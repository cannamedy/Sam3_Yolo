"""
02_convert_to_yolo.py

将 01_sam3_prelabel.py 输出的标准实例 mask
转换为 Ultralytics YOLO segmentation 数据集。

标准接口：

    data/sam3_prelabels/
        tiles/
        instances/
        metadata/
        dataset_meta.json

            ↓

    data/yolo_dataset/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml
        dataset_summary.json

重要原则：

1. 不重新切原图。
2. 不重新运行 SAM3。
3. 直接读取 01 保存的 tile + instance NPZ。
4. 多张原图时按照 source image 划分 train/val，
   避免同一张大图的相邻 tile 同时进入训练集和验证集。
"""

# =========================================================
# 0. USER CONFIG
# =========================================================

from pathlib import Path

PROJECT_ROOT = Path.cwd()

PRELABEL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "sam3_prelabels"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
)


# ---------- 数据划分 ----------
VAL_RATIO = 0.20
RANDOM_SEED = 42


# ---------- 负样本 ----------
# 负样本数量 / 正样本数量。
#
# 1.0：
#   最多保留与正样本数量相同的负样本。
#
# 2.0：
#   最多保留正样本两倍的负样本。
#
# -1：
#   保留全部负样本。

NEGATIVE_RATIO = 1.0


# ---------- Polygon ----------
MIN_INSTANCE_AREA = 3

# cv2.approxPolyDP 的 epsilon / perimeter。
# 越小轮廓越精细。
POLYGON_EPSILON_RATIO = 0.001

MAX_POLYGON_POINTS = 1000

# 极细目标无法形成有效 polygon 时，
# 是否轻微膨胀。
#
# 0 = 不膨胀
# 1 = 膨胀一次
THIN_OBJECT_DILATE = 1


CLEAR_OUTPUT = True


# =========================================================
# 1. IMPORT
# =========================================================

import random
import shutil
from collections import defaultdict

import cv2

from spot_utils import (
    clear_directory,
    load_json,
    load_instance_npz,
    instances_to_yolo_lines,
    save_json,
)


# =========================================================
# 2. PATHS
# =========================================================

TILE_DIR = PRELABEL_ROOT / "tiles"
INSTANCE_DIR = PRELABEL_ROOT / "instances"
METADATA_DIR = PRELABEL_ROOT / "metadata"

DATASET_META_PATH = (
    PRELABEL_ROOT
    / "dataset_meta.json"
)


# =========================================================
# 3. LOAD META
# =========================================================

if not DATASET_META_PATH.exists():
    raise FileNotFoundError(
        "没有找到 dataset_meta.json。\n"
        "请先运行：python 01_sam3_prelabel.py"
    )

dataset_meta = load_json(
    DATASET_META_PATH
)

classes = dataset_meta["classes"]

CLASS_NAMES = {
    int(item["id"]): str(item["name"])
    for item in classes
}


# =========================================================
# 4. RECORDS
# =========================================================

def load_records():

    metadata_files = sorted(
        METADATA_DIR.glob("*.json")
    )

    records = []

    for metadata_path in metadata_files:

        metadata = load_json(
            metadata_path
        )

        tile_stem = metadata_path.stem

        tile_path = (
            TILE_DIR
            / metadata["tile_name"]
        )

        npz_path = (
            INSTANCE_DIR
            / f"{tile_stem}.npz"
        )

        if not tile_path.exists():
            raise FileNotFoundError(
                f"缺少 tile：{tile_path}"
            )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"缺少 instance NPZ：{npz_path}"
            )

        image = cv2.imread(
            str(tile_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"无法读取：{tile_path}"
            )

        height, width = image.shape[:2]

        instance_data = load_instance_npz(
            npz_path
        )

        lines, stats = (
            instances_to_yolo_lines(
                masks=instance_data["masks"],
                class_ids=instance_data[
                    "class_ids"
                ],
                image_width=width,
                image_height=height,
                min_instance_area=(
                    MIN_INSTANCE_AREA
                ),
                epsilon_ratio=(
                    POLYGON_EPSILON_RATIO
                ),
                max_points=(
                    MAX_POLYGON_POINTS
                ),
                thin_dilate=(
                    THIN_OBJECT_DILATE
                ),
            )
        )

        records.append(
            {
                "source_id": metadata[
                    "source_id"
                ],
                "source_image": metadata[
                    "source_image"
                ],
                "tile_name": metadata[
                    "tile_name"
                ],
                "tile_path": tile_path,
                "npz_path": npz_path,
                "label_lines": lines,
                "positive": len(lines) > 0,
                "stats": stats,
            }
        )

    return records


# =========================================================
# 5. TRAIN / VAL SPLIT
# =========================================================

def assign_source_split(
    records,
    val_ratio,
    seed,
):
    sources = sorted(
        {
            record["source_id"]
            for record in records
        }
    )

    rng = random.Random(seed)

    if len(sources) >= 2:

        rng.shuffle(sources)

        val_count = max(
            1,
            int(
                round(
                    len(sources)
                    * val_ratio
                )
            ),
        )

        val_count = min(
            val_count,
            len(sources) - 1,
        )

        val_sources = set(
            sources[:val_count]
        )

        for record in records:
            record["split"] = (
                "val"
                if record["source_id"]
                in val_sources
                else "train"
            )

        return

    # 只有一张源图时，
    # 无法做到 source-level 隔离。
    #
    # 退化为 tile-level 随机划分。

    print(
        "警告：只有 1 张源图，"
        "train/val 将按 tile 划分。"
    )

    indices = list(
        range(len(records))
    )

    rng.shuffle(indices)

    val_count = max(
        1,
        int(
            round(
                len(records)
                * val_ratio
            )
        ),
    )

    if len(records) > 1:
        val_count = min(
            val_count,
            len(records) - 1,
        )

    val_indices = set(
        indices[:val_count]
    )

    for index, record in enumerate(
        records
    ):
        record["split"] = (
            "val"
            if index in val_indices
            else "train"
        )


# =========================================================
# 6. NEGATIVE SAMPLING
# =========================================================

def sample_negatives(
    records,
    ratio,
    seed,
):
    if ratio < 0:
        return records

    rng = random.Random(seed)

    selected = []

    for split in (
        "train",
        "val",
    ):

        subset = [
            x
            for x in records
            if x["split"] == split
        ]

        positives = [
            x
            for x in subset
            if x["positive"]
        ]

        negatives = [
            x
            for x in subset
            if not x["positive"]
        ]

        if positives:

            max_negative = int(
                round(
                    len(positives)
                    * ratio
                )
            )

        else:

            max_negative = min(
                len(negatives),
                10,
            )

        rng.shuffle(negatives)

        selected.extend(
            positives
        )

        selected.extend(
            negatives[
                :max_negative
            ]
        )

    return selected


# =========================================================
# 7. YAML
# =========================================================

def write_data_yaml():

    class_ids = sorted(
        CLASS_NAMES.keys()
    )

    expected_ids = list(
        range(len(class_ids))
    )

    if class_ids != expected_ids:
        raise ValueError(
            "YOLO class id 必须从 0 开始连续排列。"
        )

    lines = [
        f"path: {OUTPUT_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]

    for class_id in class_ids:

        class_name = CLASS_NAMES[
            class_id
        ]

        lines.append(
            f"  {class_id}: {class_name}"
        )

    (
        OUTPUT_ROOT
        / "data.yaml"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =========================================================
# 8. WRITE DATASET
# =========================================================

def write_dataset(records):

    summary = {
        "train": {
            "images": 0,
            "positive": 0,
            "negative": 0,
            "instances": 0,
        },
        "val": {
            "images": 0,
            "positive": 0,
            "negative": 0,
            "instances": 0,
        },
        "tiles": [],
    }

    for split in (
        "train",
        "val",
    ):

        (
            OUTPUT_ROOT
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    for record in records:

        split = record["split"]

        tile_path = record[
            "tile_path"
        ]

        tile_name = Path(
            record["tile_name"]
        ).name

        stem = Path(
            tile_name
        ).stem

        output_image = (
            OUTPUT_ROOT
            / "images"
            / split
            / tile_name
        )

        output_label = (
            OUTPUT_ROOT
            / "labels"
            / split
            / f"{stem}.txt"
        )

        shutil.copy2(
            tile_path,
            output_image,
        )

        lines = record[
            "label_lines"
        ]

        output_label.write_text(
            (
                "\n".join(lines)
                + ("\n" if lines else "")
            ),
            encoding="utf-8",
        )

        item_summary = summary[
            split
        ]

        item_summary["images"] += 1

        instance_count = len(lines)

        item_summary[
            "instances"
        ] += instance_count

        if instance_count:

            item_summary[
                "positive"
            ] += 1

        else:

            item_summary[
                "negative"
            ] += 1

        summary["tiles"].append(
            {
                "source_id": record[
                    "source_id"
                ],
                "source_image": record[
                    "source_image"
                ],
                "tile_name": tile_name,
                "split": split,
                "instances": (
                    instance_count
                ),
                **record["stats"],
            }
        )

    return summary


# =========================================================
# 9. MAIN
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

    records = load_records()

    if not records:
        raise RuntimeError(
            "01 输出目录中没有可转换的 tile。"
        )

    assign_source_split(
        records=records,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )

    before_sampling = len(
        records
    )

    positive_before = sum(
        x["positive"]
        for x in records
    )

    records = sample_negatives(
        records=records,
        ratio=NEGATIVE_RATIO,
        seed=RANDOM_SEED,
    )

    summary = write_dataset(
        records
    )

    write_data_yaml()

    final_summary = {
        "pipeline": (
            "sam3_prelabels_to_yolo_seg"
        ),

        "source_prelabels": str(
            PRELABEL_ROOT
        ),

        "output_root": str(
            OUTPUT_ROOT
        ),

        "classes": classes,

        "val_ratio": VAL_RATIO,

        "negative_ratio": (
            NEGATIVE_RATIO
        ),

        "records_before_sampling": (
            before_sampling
        ),

        "positive_before_sampling": (
            positive_before
        ),

        "negative_before_sampling": (
            before_sampling
            - positive_before
        ),

        "records_after_sampling": len(
            records
        ),

        "splits": {
            "train": summary[
                "train"
            ],
            "val": summary[
                "val"
            ],
        },

        "tiles": summary["tiles"],
    }

    save_json(
        final_summary,
        OUTPUT_ROOT
        / "dataset_summary.json",
    )

    print(
        "\n========== YOLO Dataset =========="
    )

    print(
        "Train:",
        summary["train"],
    )

    print(
        "Val  :",
        summary["val"],
    )

    print(
        "YAML :",
        OUTPUT_ROOT
        / "data.yaml",
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    main()
