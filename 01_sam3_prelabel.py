"""
01_sam3_prelabel.py

通用 SAM3 实例分割预标注脚本。

流程：
    原始大图
    -> 重叠切片
    -> SAM3 文本 Prompt 实例分割
    -> 通用 mask 后处理
    -> 同类别重复实例去重
    -> 保存 tile / instances.npz / metadata.json

本文件只负责：
    SAM3 -> 标准实例 mask

不包含：
    - 特定目标的颜色/灰度规则
    - 圆形、环形、暗斑等形态规则
    - YOLO 数据集划分
    - YOLO polygon 转换
    - YOLO 训练

更换项目时通常只需要修改 USER CONFIG 区域。
"""

# =========================================================
# 0. USER CONFIG
# =========================================================

import os

# 物理 GPU 编号。
# 如果机器只有一张卡，一般使用 "0"。
GPU_PHYSICAL_ID = os.getenv("GPU_PHYSICAL_ID", "0")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_PHYSICAL_ID
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)


# ---------- SAM3 路径 ----------
SAM3_SOURCE_PATH = os.getenv(
    "SAM3_SOURCE_PATH",
    "/software/sam3-main",
)

SAM3_MODEL_PATH = os.getenv(
    "SAM3_MODEL_PATH",
    os.path.join(SAM3_SOURCE_PATH, "sam3.pt"),
)


# ---------- 类别 ----------
# 支持多类别。
#
# 单类别示例：
#
# CLASSES = [
#     {
#         "id": 0,
#         "name": "target",
#         "prompts": [
#             "target object",
#         ],
#     },
# ]
#
# 同一个类别可以有多个 Prompt，用于提高召回。
#
# 多类别时继续增加字典即可。

CLASSES = [
    {
        "id": 0,
        "name": "target",
        "prompts": [
            "target object",
        ],
    },
]


# ---------- 切片 ----------
TILE_SIZE = 512
TILE_OVERLAP = 128


# ---------- SAM3 ----------
SAM3_MAX_SIZE = 1008

# SAM3 detection threshold。
DETECTION_THRESHOLD = 0.005

# SAM3 输出概率 mask 转二值 mask 的阈值。
MASK_THRESHOLD = 0.50


# ---------- 通用实例过滤 ----------
# 只保留最基础、跨项目通用的约束。

MIN_INSTANCE_AREA = 3

# 设为 None 表示不限制。
MAX_INSTANCE_AREA = None

# 是否保留触碰 tile 边缘的目标。
# 推荐 True，因为重叠切片会在其他 tile 中再次看到目标。
KEEP_BORDER_INSTANCES = True


# ---------- 通用形态学 ----------
# 1 表示关闭。
MORPH_OPEN_KSIZE = 1
MORPH_CLOSE_KSIZE = 1


# ---------- 重复实例 ----------
# 同类别、同 tile 内两个实例 IoU 高于此值时只保留高分实例。
DUPLICATE_IOU_THRESHOLD = 0.85


# ---------- 输出 ----------
CLEAR_OLD_OUTPUT = True

SAVE_PREVIEWS = True
MAX_PREVIEW_COUNT = 50
SAVE_PREVIEW_ONLY_IF_HAS_INSTANCE = True

PROGRESS_EVERY = 10


# =========================================================
# 1. IMPORT
# =========================================================

import gc
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from spot_utils import (
    clear_directory,
    generate_tiles,
    safe_stem,
    clean_binary_mask,
    calculate_mask_features,
    remove_duplicate_instances,
    save_instance_npz,
    save_json,
    draw_instance_preview,
)


# =========================================================
# 2. PATHS
# =========================================================

PROJECT_ROOT = Path.cwd()

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_images"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "sam3_prelabels"

TILE_DIR = OUTPUT_ROOT / "tiles"
INSTANCE_DIR = OUTPUT_ROOT / "instances"
METADATA_DIR = OUTPUT_ROOT / "metadata"
PREVIEW_DIR = OUTPUT_ROOT / "previews"

DATASET_META_PATH = OUTPUT_ROOT / "dataset_meta.json"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =========================================================
# 3. VALIDATION
# =========================================================

if not RAW_IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"原图目录不存在：{RAW_IMAGE_DIR}"
    )

sam3_source = Path(SAM3_SOURCE_PATH)
sam3_model_path = Path(SAM3_MODEL_PATH)

if not sam3_source.exists():
    raise FileNotFoundError(
        f"SAM3 源码目录不存在：{sam3_source}"
    )

if not sam3_model_path.exists():
    raise FileNotFoundError(
        f"SAM3 模型不存在：{sam3_model_path}"
    )

if str(sam3_source) not in sys.path:
    sys.path.insert(0, str(sam3_source))


def validate_classes():
    ids = []
    names = []

    for item in CLASSES:
        class_id = int(item["id"])
        name = str(item["name"])
        prompts = item.get("prompts", [])

        if not prompts:
            raise ValueError(
                f"class {class_id} 没有配置 prompts"
            )

        ids.append(class_id)
        names.append(name)

    if len(ids) != len(set(ids)):
        raise ValueError("CLASSES 中存在重复 class id")

    if len(names) != len(set(names)):
        raise ValueError("CLASSES 中存在重复 class name")


validate_classes()


# =========================================================
# 4. DEVICE
# =========================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

else:
    DEVICE = torch.device("cpu")


print("\n========== SAM3 Prelabel ==========")
print("Project root :", PROJECT_ROOT)
print("Raw images   :", RAW_IMAGE_DIR)
print("SAM3 source  :", sam3_source)
print("SAM3 model   :", sam3_model_path)
print("Device       :", DEVICE)
print("Tile size    :", TILE_SIZE)
print("Overlap      :", TILE_OVERLAP)
print("Classes      :", [(x["id"], x["name"]) for x in CLASSES])
print("===================================\n")


# =========================================================
# 5. SAM3 IMPORT
# =========================================================

from sam3 import build_sam3_image_model

from sam3.train.data.collator import (
    collate_fn_api as collate,
)

from sam3.model.utils.misc import (
    copy_data_to_device,
)

from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)

from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)

from sam3.eval.postprocessors import (
    PostProcessImage,
)


# =========================================================
# 6. LOAD MODEL
# =========================================================

gc.collect()

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

sam3_model = build_sam3_image_model(
    checkpoint_path=str(sam3_model_path),
    device=DEVICE,
    eval_mode=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
)

sam3_model = sam3_model.to(DEVICE)
sam3_model.eval()


transform = ComposeAPI(
    transforms=[
        RandomResizeAPI(
            sizes=SAM3_MAX_SIZE,
            max_size=SAM3_MAX_SIZE,
            square=True,
            consistent_transform=False,
        ),
        ToTensorAPI(),
        NormalizeAPI(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ]
)


postprocessor = PostProcessImage(
    max_dets_per_img=-1,
    iou_type="segm",
    use_original_sizes_box=True,
    use_original_sizes_mask=True,
    convert_mask_to_rle=False,
    detection_threshold=float(DETECTION_THRESHOLD),
    to_cpu=DEVICE.type != "cuda",
)


# =========================================================
# 7. SAM3 HELPERS
# =========================================================

GLOBAL_QUERY_ID = 1


def create_datapoint(pil_image, prompt):
    global GLOBAL_QUERY_ID

    width, height = pil_image.size

    datapoint = Datapoint(
        find_queries=[],
        images=[
            SAMImage(
                data=pil_image,
                objects=[],
                size=[height, width],
            )
        ],
    )

    query_id = GLOBAL_QUERY_ID
    GLOBAL_QUERY_ID += 1

    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=prompt,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=query_id,
                original_image_id=query_id,
                original_category_id=1,
                original_size=[width, height],
                object_id=0,
                frame_index=0,
            ),
        )
    )

    return datapoint


def normalize_model_outputs(outputs):
    if isinstance(outputs, dict):
        return outputs

    if isinstance(outputs, (list, tuple)):
        if not outputs:
            raise RuntimeError("SAM3 outputs 为空")

        return normalize_model_outputs(outputs[-1])

    result = {}

    for key in (
        "pred_logits",
        "pred_boxes",
        "pred_masks",
    ):
        if hasattr(outputs, key):
            value = getattr(outputs, key)

            if value is not None:
                result[key] = value

    if result:
        return result

    if hasattr(outputs, "output"):
        return normalize_model_outputs(outputs.output)

    raise TypeError(
        f"无法解析 SAM3 输出：{type(outputs)}"
    )


def split_batch_predictions(predictions, batch_size):
    if isinstance(predictions, list):
        if len(predictions) != batch_size:
            raise RuntimeError(
                f"prediction 数量不匹配："
                f"{len(predictions)} != {batch_size}"
            )

        return predictions

    if not isinstance(predictions, dict):
        raise TypeError(
            f"未知 prediction 类型：{type(predictions)}"
        )

    result = []

    for index in range(batch_size):
        item = {}

        for key, value in predictions.items():

            if (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                item[key] = value[index]

            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                item[key] = value[index]

            else:
                item[key] = value

        result.append(item)

    return result


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def extract_masks_and_scores(
    prediction,
    target_height,
    target_width,
):
    if prediction is None:
        return [], []

    masks_value = None

    if isinstance(prediction, dict):
        masks_value = prediction.get("masks")

        if masks_value is None:
            masks_value = prediction.get("segm")

    if masks_value is None:
        return [], []

    masks = to_numpy(masks_value)

    if masks.ndim == 2:
        masks = masks[None, ...]

    elif masks.ndim == 4:
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise RuntimeError(
            f"不支持的 mask shape：{masks.shape}"
        )

    scores_value = prediction.get("scores")

    if scores_value is None:
        scores = np.zeros(
            len(masks),
            dtype=np.float32,
        )
    else:
        scores = to_numpy(
            scores_value
        ).reshape(-1)

    if len(scores) < len(masks):
        scores = np.pad(
            scores,
            (0, len(masks) - len(scores)),
        )

    probability_masks = []

    for mask in masks:
        mask = mask.astype(np.float32)

        if mask.shape != (
            target_height,
            target_width,
        ):
            mask = cv2.resize(
                mask,
                (
                    target_width,
                    target_height,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

        probability_masks.append(mask)

    return (
        probability_masks,
        [
            float(x)
            for x in scores[:len(probability_masks)]
        ],
    )


# =========================================================
# 8. PROMPT TABLE
# =========================================================

PROMPT_TABLE = []

for class_item in CLASSES:

    class_id = int(class_item["id"])
    class_name = str(class_item["name"])

    for prompt in class_item["prompts"]:
        PROMPT_TABLE.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "prompt": str(prompt),
            }
        )


CLASS_NAME_MAP = {
    int(item["id"]): str(item["name"])
    for item in CLASSES
}


# =========================================================
# 9. TILE INFERENCE
# =========================================================

def infer_tile(tile_bgr):
    tile_rgb = cv2.cvtColor(
        tile_bgr,
        cv2.COLOR_BGR2RGB,
    )

    pil_image = Image.fromarray(tile_rgb)

    datapoints = []

    for prompt_item in PROMPT_TABLE:

        datapoint = create_datapoint(
            pil_image,
            prompt_item["prompt"],
        )

        datapoint = transform(datapoint)

        datapoints.append(datapoint)

    batch = collate(
        datapoints,
        dict_key="data",
    )["data"]

    batch = copy_data_to_device(
        batch,
        DEVICE,
        non_blocking=True,
    )

    with torch.inference_mode():
        outputs = sam3_model(batch)

    outputs = normalize_model_outputs(outputs)

    batch_size = len(PROMPT_TABLE)

    target_sizes = torch.tensor(
        [[TILE_SIZE, TILE_SIZE]] * batch_size,
        dtype=torch.int64,
        device=DEVICE,
    )

    predictions = postprocessor(
        outputs,
        target_sizes_boxes=target_sizes,
        target_sizes_masks=target_sizes,
    )

    prediction_list = split_batch_predictions(
        predictions,
        batch_size=batch_size,
    )

    instances = []

    for prompt_id, prediction in enumerate(
        prediction_list
    ):

        prompt_item = PROMPT_TABLE[prompt_id]

        masks, scores = extract_masks_and_scores(
            prediction,
            target_height=TILE_SIZE,
            target_width=TILE_SIZE,
        )

        for mask_index, probability_mask in enumerate(
            masks
        ):

            binary_mask = (
                probability_mask
                >= float(MASK_THRESHOLD)
            ).astype(np.uint8) * 255

            binary_mask = clean_binary_mask(
                binary_mask,
                min_area=MIN_INSTANCE_AREA,
                max_area=MAX_INSTANCE_AREA,
                open_ksize=MORPH_OPEN_KSIZE,
                close_ksize=MORPH_CLOSE_KSIZE,
            )

            if not np.any(binary_mask):
                continue

            features = calculate_mask_features(
                binary_mask
            )

            if (
                not KEEP_BORDER_INSTANCES
                and features["border_touch"]
            ):
                continue

            score = (
                float(scores[mask_index])
                if mask_index < len(scores)
                else 0.0
            )

            instances.append(
                {
                    "mask": binary_mask,
                    "score": score,
                    "class_id": prompt_item[
                        "class_id"
                    ],
                    "class_name": prompt_item[
                        "class_name"
                    ],
                    "prompt_id": prompt_id,
                    "prompt": prompt_item[
                        "prompt"
                    ],
                    "features": features,
                }
            )

    instances = remove_duplicate_instances(
        instances,
        iou_threshold=DUPLICATE_IOU_THRESHOLD,
        compare_within_class=True,
    )

    return instances


# =========================================================
# 10. FILE HELPERS
# =========================================================

def find_raw_images():
    return sorted(
        [
            path
            for path in RAW_IMAGE_DIR.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ],
        key=lambda path: path.name,
    )


# =========================================================
# 11. MAIN
# =========================================================

def main():

    if CLEAR_OLD_OUTPUT:
        clear_directory(OUTPUT_ROOT)

    for directory in (
        TILE_DIR,
        INSTANCE_DIR,
        METADATA_DIR,
        PREVIEW_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    source_images = find_raw_images()

    if not source_images:
        raise RuntimeError(
            f"没有找到原始图片：{RAW_IMAGE_DIR}"
        )

    dataset_meta = {
        "pipeline": "sam3_to_yolo_seg",
        "tile_size": TILE_SIZE,
        "tile_overlap": TILE_OVERLAP,
        "mask_threshold": MASK_THRESHOLD,
        "classes": CLASSES,
        "sources": [],
    }

    total_tiles = 0
    total_instances = 0
    preview_count = 0

    for source_index, image_path in enumerate(
        source_images
    ):

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            print(
                f"跳过无法读取图片：{image_path}"
            )
            continue

        height, width = image.shape[:2]

        source_id = (
            f"{source_index:04d}_"
            f"{safe_stem(image_path.stem)}"
        )

        tiles = generate_tiles(
            image=image,
            tile_size=TILE_SIZE,
            overlap=TILE_OVERLAP,
            pad_edge=True,
            pad_value=128,
            source_name=source_id,
        )

        source_meta = {
            "source_id": source_id,
            "source_image": image_path.name,
            "width": width,
            "height": height,
            "tile_count": len(tiles),
        }

        dataset_meta["sources"].append(
            source_meta
        )

        print(
            f"\n[{source_index + 1}/"
            f"{len(source_images)}] "
            f"{image_path.name}"
        )

        for tile_index, tile_info in enumerate(
            tiles
        ):

            tile = tile_info["tile"]

            tile_name = (
                f"{source_id}"
                f"__x{tile_info['x1']:06d}"
                f"_y{tile_info['y1']:06d}.png"
            )

            tile_stem = Path(
                tile_name
            ).stem

            instances = infer_tile(tile)

            total_tiles += 1
            total_instances += len(instances)

            tile_path = (
                TILE_DIR
                / tile_name
            )

            cv2.imwrite(
                str(tile_path),
                tile,
            )

            npz_path = (
                INSTANCE_DIR
                / f"{tile_stem}.npz"
            )

            save_instance_npz(
                save_path=npz_path,
                masks=[
                    x["mask"]
                    for x in instances
                ],
                scores=[
                    x["score"]
                    for x in instances
                ],
                class_ids=[
                    x["class_id"]
                    for x in instances
                ],
                prompt_ids=[
                    x["prompt_id"]
                    for x in instances
                ],
                image_shape=tile.shape[:2],
            )

            metadata = {
                "source_id": source_id,
                "source_image": image_path.name,
                "source_width": width,
                "source_height": height,

                "tile_name": tile_name,
                "tile_index": tile_index,

                "x1": tile_info["x1"],
                "y1": tile_info["y1"],
                "x2": tile_info["x2"],
                "y2": tile_info["y2"],

                "valid_w": tile_info["valid_w"],
                "valid_h": tile_info["valid_h"],

                "instance_count": len(
                    instances
                ),

                "instances": [
                    {
                        "score": item[
                            "score"
                        ],
                        "class_id": item[
                            "class_id"
                        ],
                        "class_name": item[
                            "class_name"
                        ],
                        "prompt_id": item[
                            "prompt_id"
                        ],
                        "prompt": item[
                            "prompt"
                        ],
                        "features": item[
                            "features"
                        ],
                    }
                    for item in instances
                ],
            }

            save_json(
                metadata,
                METADATA_DIR
                / f"{tile_stem}.json",
            )

            save_preview = (
                SAVE_PREVIEWS
                and preview_count
                < MAX_PREVIEW_COUNT
                and (
                    not SAVE_PREVIEW_ONLY_IF_HAS_INSTANCE
                    or len(instances) > 0
                )
            )

            if save_preview:

                preview = draw_instance_preview(
                    image=tile,
                    instances=instances,
                    class_names=CLASS_NAME_MAP,
                )

                cv2.imwrite(
                    str(
                        PREVIEW_DIR
                        / f"{tile_stem}.jpg"
                    ),
                    preview,
                )

                preview_count += 1

            if (
                total_tiles
                % PROGRESS_EVERY
                == 0
            ):
                print(
                    f"  tiles={total_tiles}, "
                    f"instances="
                    f"{total_instances}"
                )

    dataset_meta[
        "total_tiles"
    ] = total_tiles

    dataset_meta[
        "total_instances"
    ] = total_instances

    save_json(
        dataset_meta,
        DATASET_META_PATH,
    )

    print(
        "\n========== 完成 =========="
    )

    print(
        "Source images :",
        len(dataset_meta["sources"]),
    )

    print(
        "Tiles         :",
        total_tiles,
    )

    print(
        "Instances     :",
        total_instances,
    )

    print(
        "Output        :",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
