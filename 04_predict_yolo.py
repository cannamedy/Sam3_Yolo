"""
04_predict_yolo.py

通用 YOLO 实例分割大图推理脚本。

流程：

    原始图片
    -> 重叠切片
    -> YOLO segmentation
    -> tile mask 恢复至原图坐标
    -> overlap 区域并集
    -> 输出各类别整图 mask
    -> 输出 combined mask
    -> 输出 overlay
    -> 输出 summary.json

输入模型：
    03_train_yolo.py 生成的 best.pt

本文件完全不依赖 SAM3。
"""

# =========================================================
# 0. USER CONFIG
# =========================================================

from pathlib import Path

PROJECT_ROOT = Path.cwd()


# ---------- 待预测图片 ----------
INFERENCE_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "inference_images"
)

# inference_images 为空时可回退到 raw_images。
RAW_IMAGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw_images"
)


# ---------- 模型 ----------
# None：
# 自动查找 runs_yolo 下最新 best.pt。
#
# 也可以直接填写：
#
# MODEL_OVERRIDE = "/path/to/best.pt"

MODEL_OVERRIDE = None


# ---------- 输出 ----------
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "yolo_predictions"
)


# ---------- 切片 ----------
TILE_SIZE = 512
TILE_OVERLAP = 128


# ---------- YOLO ----------
IMAGE_SIZE = 512

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.70

DEVICE = "0"


# ---------- Mask ----------
MASK_THRESHOLD = 0.50


CLEAR_OUTPUT = True


# =========================================================
# 1. IMPORT
# =========================================================

import gc

import cv2
import numpy as np

from ultralytics import YOLO

from spot_utils import (
    clear_directory,
    list_image_files,
    generate_tiles,
    safe_stem,
    restore_mask_to_full_image,
    create_class_overlay,
    save_json,
)


# =========================================================
# 2. MODEL
# =========================================================

def find_best_pt():

    if MODEL_OVERRIDE is not None:

        path = Path(
            MODEL_OVERRIDE
        ).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"MODEL_OVERRIDE 不存在：{path}"
            )

        return path.resolve()

    preferred = (
        PROJECT_ROOT
        / "runs_yolo"
        / "yolo_seg"
        / "weights"
        / "best.pt"
    )

    if preferred.exists():
        return preferred

    candidates = list(
        (
            PROJECT_ROOT
            / "runs_yolo"
        ).glob(
            "**/weights/best.pt"
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "没有找到 best.pt。\n"
            "请先运行："
            "python 03_train_yolo.py"
        )

    return max(
        candidates,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )


# =========================================================
# 3. INPUT
# =========================================================

def choose_input_images():

    images = list_image_files(
        INFERENCE_IMAGE_DIR
    )

    if images:
        return [
            Path(x)
            for x in images
        ]

    images = list_image_files(
        RAW_IMAGE_DIR
    )

    if images:
        return [
            Path(x)
            for x in images
        ]

    raise FileNotFoundError(
        "没有找到待预测图片。\n"
        f"优先目录：{INFERENCE_IMAGE_DIR}\n"
        f"回退目录：{RAW_IMAGE_DIR}"
    )


# =========================================================
# 4. MASK EXTRACTION
# =========================================================

def extract_result_instances(
    result,
    tile_height,
    tile_width,
):

    if (
        result.masks is None
        or result.boxes is None
    ):
        return []

    mask_tensor = (
        result
        .masks
        .data
    )

    if mask_tensor is None:
        return []

    masks = (
        mask_tensor
        .detach()
        .cpu()
        .numpy()
    )

    class_ids = (
        result
        .boxes
        .cls
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    confidences = (
        result
        .boxes
        .conf
        .detach()
        .cpu()
        .numpy()
    )

    instances = []

    count = min(
        len(masks),
        len(class_ids),
        len(confidences),
    )

    for index in range(count):

        mask = masks[index]

        if mask.shape != (
            tile_height,
            tile_width,
        ):

            mask = cv2.resize(
                mask.astype(
                    np.float32
                ),
                (
                    tile_width,
                    tile_height,
                ),
                interpolation=(
                    cv2.INTER_LINEAR
                ),
            )

        binary = (
            mask
            >= MASK_THRESHOLD
        ).astype(
            np.uint8
        ) * 255

        if not np.any(binary):
            continue

        instances.append(
            {
                "mask": binary,
                "class_id": int(
                    class_ids[index]
                ),
                "confidence": float(
                    confidences[index]
                ),
            }
        )

    return instances


# =========================================================
# 5. PREDICT ONE IMAGE
# =========================================================

def predict_image(
    model,
    image_path,
):

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{image_path}"
        )

    full_height, full_width = (
        image.shape[:2]
    )

    image_id = safe_stem(
        image_path.stem
    )

    output_dir = (
        OUTPUT_ROOT
        / image_id
    )

    class_mask_dir = (
        output_dir
        / "class_masks"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tiles = generate_tiles(
        image=image,
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP,
        pad_edge=True,
        pad_value=128,
        source_name=image_id,
    )

    class_masks = {}

    raw_instance_count = 0

    confidence_values = []

    for tile_info in tiles:

        tile = tile_info[
            "tile"
        ]

        result_list = model.predict(
            source=tile,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        if not result_list:
            continue

        result = result_list[0]

        instances = (
            extract_result_instances(
                result=result,
                tile_height=(
                    tile.shape[0]
                ),
                tile_width=(
                    tile.shape[1]
                ),
            )
        )

        raw_instance_count += len(
            instances
        )

        for instance in instances:

            class_id = instance[
                "class_id"
            ]

            confidence_values.append(
                instance[
                    "confidence"
                ]
            )

            local_mask = instance[
                "mask"
            ]

            restored = (
                restore_mask_to_full_image(
                    tile_mask=local_mask,
                    x1=tile_info["x1"],
                    y1=tile_info["y1"],
                    full_height=(
                        full_height
                    ),
                    full_width=(
                        full_width
                    ),
                    valid_w=tile_info[
                        "valid_w"
                    ],
                    valid_h=tile_info[
                        "valid_h"
                    ],
                    dtype=np.uint8,
                )
            )

            if class_id not in class_masks:

                class_masks[
                    class_id
                ] = np.zeros(
                    (
                        full_height,
                        full_width,
                    ),
                    dtype=np.uint8,
                )

            class_masks[
                class_id
            ] = np.maximum(
                class_masks[class_id],
                restored,
            )

    combined_mask = np.zeros(
        (
            full_height,
            full_width,
        ),
        dtype=np.uint8,
    )

    for class_id, mask in (
        class_masks.items()
    ):

        combined_mask = np.maximum(
            combined_mask,
            mask,
        )

        class_name = (
            model.names.get(
                class_id,
                f"class_{class_id}",
            )
            if isinstance(
                model.names,
                dict,
            )
            else f"class_{class_id}"
        )

        safe_name = safe_stem(
            str(class_name)
        )

        cv2.imwrite(
            str(
                class_mask_dir
                / (
                    f"{class_id:02d}_"
                    f"{safe_name}.png"
                )
            ),
            mask,
        )

    combined_mask_path = (
        output_dir
        / "combined_mask.png"
    )

    cv2.imwrite(
        str(combined_mask_path),
        combined_mask,
    )

    overlay = create_class_overlay(
        image=image,
        class_masks=class_masks,
    )

    overlay_path = (
        output_dir
        / "overlay.jpg"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay,
    )

    total_pixels = (
        full_width
        * full_height
    )

    predicted_pixels = int(
        np.count_nonzero(
            combined_mask
        )
    )

    coverage_ratio = (
        100.0
        * predicted_pixels
        / max(
            total_pixels,
            1,
        )
    )

    class_summary = {}

    for class_id, mask in (
        class_masks.items()
    ):

        pixel_count = int(
            np.count_nonzero(mask)
        )

        class_name = (
            model.names.get(
                class_id,
                f"class_{class_id}",
            )
            if isinstance(
                model.names,
                dict,
            )
            else f"class_{class_id}"
        )

        class_summary[
            str(class_id)
        ] = {
            "name": str(
                class_name
            ),
            "pixels": pixel_count,
            "coverage_percent": (
                100.0
                * pixel_count
                / max(
                    total_pixels,
                    1,
                )
            ),
        }

    summary = {
        "image": image_path.name,

        "width": full_width,
        "height": full_height,

        "tile_size": TILE_SIZE,
        "tile_overlap": (
            TILE_OVERLAP
        ),
        "tile_count": len(tiles),

        # overlap tile 中同一个物体可能出现多次，
        # 因此这里只称为 raw_tile_instances。
        "raw_tile_instances": (
            raw_instance_count
        ),

        "confidence": {
            "count": len(
                confidence_values
            ),
            "min": (
                min(confidence_values)
                if confidence_values
                else None
            ),
            "max": (
                max(confidence_values)
                if confidence_values
                else None
            ),
            "mean": (
                float(
                    np.mean(
                        confidence_values
                    )
                )
                if confidence_values
                else None
            ),
        },

        "predicted_pixels": (
            predicted_pixels
        ),

        "coverage_percent": (
            coverage_ratio
        ),

        "classes": class_summary,
    }

    save_json(
        summary,
        output_dir
        / "summary.json",
    )

    print(
        f"{image_path.name}: "
        f"coverage="
        f"{coverage_ratio:.3f}%"
    )

    return summary


# =========================================================
# 6. MAIN
# =========================================================

def main():

    if CLEAR_OUTPUT:
        clear_directory(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_pt = find_best_pt()

    images = choose_input_images()

    print(
        "\n========== YOLO Seg Predict =========="
    )

    print(
        "Model :",
        best_pt,
    )

    print(
        "Images:",
        len(images),
    )

    print(
        "Output:",
        OUTPUT_ROOT,
    )

    print(
        "======================================\n"
    )

    model = YOLO(
        str(best_pt)
    )

    all_summaries = []

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        summary = predict_image(
            model,
            image_path,
        )

        all_summaries.append(
            summary
        )

    save_json(
        {
            "model": str(
                best_pt
            ),
            "image_count": len(
                all_summaries
            ),
            "images": all_summaries,
        },
        OUTPUT_ROOT
        / "prediction_summary.json",
    )

    print(
        "\n预测完成：",
        OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
