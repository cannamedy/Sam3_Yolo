"""
spot_utils.py

SAM3 -> YOLO Segmentation 基线工程公共函数。

职责：

1. 文件与 JSON 工具
2. 图像重叠切片
3. 二值 mask 基础处理
4. mask 几何特征
5. instance IoU 与去重
6. instance NPZ 保存 / 读取
7. mask -> YOLO segmentation polygon
8. tile mask -> 原图坐标恢复
9. 基础可视化

本文件不得包含具体项目目标规则，例如：

- 某种颜色阈值
- 某种灰度阈值
- 圆形 / 环形规则
- 暗斑规则
- 某一项目固定面积阈值
- 某一业务专用 presentation

项目特有逻辑应放在项目脚本或独立模块中。
"""

from __future__ import annotations

import json
import os
import re
import shutil

from pathlib import Path

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cv2
import numpy as np


PathLike = Union[
    str,
    os.PathLike,
]


# =========================================================
# 0. FILE UTILITIES
# =========================================================

def ensure_dir(
    path: PathLike,
) -> Path:

    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def clear_directory(
    path: PathLike,
) -> None:

    path = Path(path)

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_stem(
    value: str,
) -> str:

    value = re.sub(
        r"[^0-9a-zA-Z_-]+",
        "_",
        str(value),
    ).strip("_")

    return value or "item"


def list_image_files(
    image_dir: PathLike,
    recursive: bool = False,
    extensions: Sequence[str] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ),
) -> List[str]:

    image_dir = Path(
        image_dir
    )

    if not image_dir.exists():
        return []

    extension_set = {
        x.lower()
        for x in extensions
    }

    iterator = (
        image_dir.rglob("*")
        if recursive
        else image_dir.iterdir()
    )

    files = [
        str(path)
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower()
            in extension_set
        )
    ]

    return sorted(files)


def _json_default(
    value: Any,
) -> Any:

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        value = float(value)

        return (
            value
            if np.isfinite(value)
            else None
        )

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    if isinstance(
        value,
        Path,
    ):
        return str(value)

    raise TypeError(
        f"无法 JSON 序列化："
        f"{type(value).__name__}"
    )


def save_json(
    data: Any,
    save_path: PathLike,
    indent: int = 2,
) -> None:

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with save_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=indent,
            default=_json_default,
            allow_nan=False,
        )


def load_json(
    path: PathLike,
) -> Any:

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =========================================================
# 1. BASIC IMAGE / MASK
# =========================================================

def ensure_odd(
    value: int,
) -> int:

    value = max(
        1,
        int(value),
    )

    return (
        value
        if value % 2 == 1
        else value + 1
    )


def as_binary_mask(
    mask: np.ndarray,
) -> np.ndarray:

    if mask is None:
        raise ValueError(
            "mask 不能为空"
        )

    return (
        np.asarray(mask) > 0
    ).astype(
        np.uint8
    ) * 255


def clean_binary_mask(
    mask: np.ndarray,
    min_area: int = 1,
    max_area: Optional[int] = None,
    open_ksize: int = 1,
    close_ksize: int = 1,
) -> np.ndarray:

    binary = as_binary_mask(
        mask
    )

    open_ksize = ensure_odd(
        open_ksize
    )

    close_ksize = ensure_odd(
        close_ksize
    )

    if open_ksize > 1:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                open_ksize,
                open_ksize,
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
        )

    if close_ksize > 1:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                close_ksize,
                close_ksize,
            ),
        )

        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
        )

    count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            (binary > 0).astype(
                np.uint8
            ),
            connectivity=8,
        )
    )

    output = np.zeros_like(
        binary,
        dtype=np.uint8,
    )

    for component_id in range(
        1,
        count,
    ):

        area = int(
            stats[
                component_id,
                cv2.CC_STAT_AREA,
            ]
        )

        if area < int(
            min_area
        ):
            continue

        if (
            max_area is not None
            and area
            > int(max_area)
        ):
            continue

        output[
            labels == component_id
        ] = 255

    return output


# =========================================================
# 2. TILING
# =========================================================

def generate_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 128,
    pad_edge: bool = True,
    pad_value: Union[
        int,
        Tuple[int, int, int],
    ] = 128,
    source_name: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if (
        image is None
        or image.size == 0
    ):
        raise ValueError(
            "输入图像为空"
        )

    tile_size = int(
        tile_size
    )

    overlap = int(
        overlap
    )

    if tile_size <= 0:
        raise ValueError(
            "tile_size 必须 > 0"
        )

    if (
        overlap < 0
        or overlap >= tile_size
    ):
        raise ValueError(
            "overlap 必须满足 "
            "0 <= overlap < tile_size"
        )

    height, width = (
        image.shape[:2]
    )

    stride = (
        tile_size
        - overlap
    )

    x_positions = list(
        range(
            0,
            max(
                width - tile_size,
                0,
            ) + 1,
            stride,
        )
    )

    y_positions = list(
        range(
            0,
            max(
                height - tile_size,
                0,
            ) + 1,
            stride,
        )
    )

    last_x = max(
        width - tile_size,
        0,
    )

    last_y = max(
        height - tile_size,
        0,
    )

    if (
        not x_positions
        or x_positions[-1]
        != last_x
    ):
        x_positions.append(
            last_x
        )

    if (
        not y_positions
        or y_positions[-1]
        != last_y
    ):
        y_positions.append(
            last_y
        )

    x_positions = sorted(
        set(x_positions)
    )

    y_positions = sorted(
        set(y_positions)
    )

    source_stem = (
        safe_stem(source_name)
        if source_name
        else "image"
    )

    tiles = []

    tile_index = 0

    for y1 in y_positions:

        for x1 in x_positions:

            x2 = min(
                x1 + tile_size,
                width,
            )

            y2 = min(
                y1 + tile_size,
                height,
            )

            crop = image[
                y1:y2,
                x1:x2,
            ]

            valid_h, valid_w = (
                crop.shape[:2]
            )

            if (
                pad_edge
                and (
                    valid_h != tile_size
                    or valid_w != tile_size
                )
            ):

                if image.ndim == 2:

                    tile = np.full(
                        (
                            tile_size,
                            tile_size,
                        ),
                        pad_value,
                        dtype=image.dtype,
                    )

                else:

                    if isinstance(
                        pad_value,
                        tuple,
                    ):

                        tile = np.zeros(
                            (
                                tile_size,
                                tile_size,
                                image.shape[2],
                            ),
                            dtype=image.dtype,
                        )

                        tile[:] = (
                            pad_value
                        )

                    else:

                        tile = np.full(
                            (
                                tile_size,
                                tile_size,
                                image.shape[2],
                            ),
                            pad_value,
                            dtype=image.dtype,
                        )

                tile[
                    :valid_h,
                    :valid_w,
                ] = crop

            else:

                tile = crop.copy()

            tile_name = (
                f"{source_stem}"
                f"__x{x1:06d}"
                f"_y{y1:06d}.png"
            )

            tiles.append(
                {
                    "tile": tile,
                    "tile_index": (
                        tile_index
                    ),
                    "tile_name": (
                        tile_name
                    ),
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "valid_w": int(
                        valid_w
                    ),
                    "valid_h": int(
                        valid_h
                    ),
                    "source_width": int(
                        width
                    ),
                    "source_height": int(
                        height
                    ),
                }
            )

            tile_index += 1

    return tiles


# =========================================================
# 3. MASK FEATURES
# =========================================================

def calculate_mask_features(
    mask: np.ndarray,
) -> Dict[str, Any]:

    binary = as_binary_mask(
        mask
    )

    area = int(
        np.count_nonzero(binary)
    )

    if area == 0:

        return {
            "area": 0,
            "bbox": None,
            "width": 0,
            "height": 0,
            "center_x": None,
            "center_y": None,
            "perimeter": 0.0,
            "circularity": 0.0,
            "solidity": 0.0,
            "extent": 0.0,
            "aspect_ratio": None,
            "equivalent_diameter": 0.0,
            "border_touch": False,
        }

    contours, _ = (
        cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
    )

    if not contours:
        raise RuntimeError(
            "mask 非空但没有找到 contour"
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    x, y, width, height = (
        cv2.boundingRect(
            contour
        )
    )

    x2 = x + width
    y2 = y + height

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    circularity = (
        float(
            4.0
            * np.pi
            * area
            / (
                perimeter
                * perimeter
            )
        )
        if perimeter > 0
        else 0.0
    )

    hull = cv2.convexHull(
        contour
    )

    hull_area = float(
        cv2.contourArea(
            hull
        )
    )

    solidity = (
        float(
            area
            / hull_area
        )
        if hull_area > 0
        else 0.0
    )

    extent = (
        float(
            area
            / max(
                width * height,
                1,
            )
        )
    )

    short_side = max(
        min(
            width,
            height,
        ),
        1,
    )

    aspect_ratio = float(
        max(
            width,
            height,
        )
        / short_side
    )

    moments = cv2.moments(
        binary,
        binaryImage=True,
    )

    if moments[
        "m00"
    ] > 0:

        center_x = float(
            moments["m10"]
            / moments["m00"]
        )

        center_y = float(
            moments["m01"]
            / moments["m00"]
        )

    else:

        center_x = float(
            x + width / 2
        )

        center_y = float(
            y + height / 2
        )

    image_h, image_w = (
        binary.shape[:2]
    )

    border_touch = bool(
        x <= 0
        or y <= 0
        or x2 >= image_w
        or y2 >= image_h
    )

    equivalent_diameter = float(
        np.sqrt(
            4.0
            * area
            / np.pi
        )
    )

    return {
        "area": area,

        "bbox": [
            int(x),
            int(y),
            int(x2),
            int(y2),
        ],

        "width": int(width),

        "height": int(height),

        "center_x": center_x,

        "center_y": center_y,

        "perimeter": perimeter,

        "circularity": circularity,

        "solidity": solidity,

        "extent": extent,

        "aspect_ratio": (
            aspect_ratio
        ),

        "equivalent_diameter": (
            equivalent_diameter
        ),

        "border_touch": (
            border_touch
        ),
    }


# =========================================================
# 4. INSTANCE IOU / DEDUPLICATION
# =========================================================

def mask_iou(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:

    if (
        mask_a.shape[:2]
        != mask_b.shape[:2]
    ):
        raise ValueError(
            "两个 mask 尺寸不一致"
        )

    a = mask_a > 0
    b = mask_b > 0

    intersection = np.count_nonzero(
        a & b
    )

    union = np.count_nonzero(
        a | b
    )

    if union == 0:
        return 0.0

    return float(
        intersection / union
    )


def remove_duplicate_instances(
    instances: Sequence[
        Dict[str, Any]
    ],
    iou_threshold: float = 0.85,
    compare_within_class: bool = True,
) -> List[Dict[str, Any]]:

    ordered = sorted(
        instances,
        key=lambda item: float(
            item.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )

    kept = []

    for candidate in ordered:

        duplicate = False

        for existing in kept:

            if (
                compare_within_class
                and int(
                    candidate.get(
                        "class_id",
                        -1,
                    )
                )
                != int(
                    existing.get(
                        "class_id",
                        -1,
                    )
                )
            ):
                continue

            iou = mask_iou(
                candidate["mask"],
                existing["mask"],
            )

            if (
                iou
                >= float(
                    iou_threshold
                )
            ):

                duplicate = True
                break

        if not duplicate:
            kept.append(
                candidate
            )

    return kept


# =========================================================
# 5. INSTANCE NPZ
# =========================================================

def save_instance_npz(
    save_path: PathLike,
    masks: Sequence[np.ndarray],
    scores: Sequence[float],
    class_ids: Sequence[int],
    prompt_ids: Optional[
        Sequence[int]
    ] = None,
    image_shape: Optional[
        Tuple[int, int]
    ] = None,
) -> None:

    save_path = Path(
        save_path
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not (
        len(masks)
        == len(scores)
        == len(class_ids)
    ):
        raise ValueError(
            "masks / scores / class_ids "
            "数量必须一致"
        )

    if prompt_ids is None:

        prompt_ids = [
            -1
        ] * len(masks)

    if (
        len(prompt_ids)
        != len(masks)
    ):
        raise ValueError(
            "prompt_ids 数量不一致"
        )

    if masks:

        mask_array = np.stack(
            [
                as_binary_mask(mask)
                for mask in masks
            ],
            axis=0,
        ).astype(
            np.uint8
        )

    else:

        if image_shape is None:

            mask_array = np.zeros(
                (
                    0,
                    0,
                    0,
                ),
                dtype=np.uint8,
            )

        else:

            height, width = (
                image_shape
            )

            mask_array = np.zeros(
                (
                    0,
                    int(height),
                    int(width),
                ),
                dtype=np.uint8,
            )

    np.savez_compressed(
        save_path,

        masks=mask_array,

        scores=np.asarray(
            scores,
            dtype=np.float32,
        ),

        class_ids=np.asarray(
            class_ids,
            dtype=np.int32,
        ),

        prompt_ids=np.asarray(
            prompt_ids,
            dtype=np.int32,
        ),
    )


def load_instance_npz(
    path: PathLike,
) -> Dict[str, np.ndarray]:

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        masks = np.asarray(
            data["masks"]
        )

        scores = np.asarray(
            data["scores"]
        )

        class_ids = np.asarray(
            data["class_ids"]
        )

        if "prompt_ids" in data:

            prompt_ids = np.asarray(
                data["prompt_ids"]
            )

        else:

            prompt_ids = np.full(
                len(masks),
                -1,
                dtype=np.int32,
            )

    return {
        "masks": masks,
        "scores": scores,
        "class_ids": class_ids,
        "prompt_ids": prompt_ids,
    }


# =========================================================
# 6. MASK -> YOLO POLYGON
# =========================================================

def reduce_polygon_points(
    points: np.ndarray,
    max_points: int,
) -> np.ndarray:

    if (
        max_points <= 0
        or len(points)
        <= max_points
    ):
        return points

    indices = np.linspace(
        0,
        len(points) - 1,
        max_points,
        dtype=np.int32,
    )

    return points[
        indices
    ]


def mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.001,
    max_points: int = 1000,
    thin_dilate: int = 1,
) -> Optional[np.ndarray]:

    work = as_binary_mask(
        mask
    )

    def largest_contour(
        current_mask,
    ):

        contours, _ = (
            cv2.findContours(
                current_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
        )

        if not contours:
            return None

        return max(
            contours,
            key=cv2.contourArea,
        )

    contour = largest_contour(
        work
    )

    if contour is None:
        return None

    if (
        len(contour) < 3
        or cv2.contourArea(
            contour
        ) < 0.5
    ):

        if thin_dilate > 0:

            kernel = (
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (3, 3),
                )
            )

            work = cv2.dilate(
                work,
                kernel,
                iterations=int(
                    thin_dilate
                ),
            )

            contour = (
                largest_contour(
                    work
                )
            )

    if contour is None:
        return None

    perimeter = float(
        cv2.arcLength(
            contour,
            True,
        )
    )

    epsilon = max(
        0.0,
        float(
            epsilon_ratio
        )
        * perimeter,
    )

    polygon = (
        cv2.approxPolyDP(
            contour,
            epsilon,
            True,
        )
        .reshape(
            -1,
            2,
        )
    )

    if len(polygon) < 3:

        x, y, width, height = (
            cv2.boundingRect(
                work
            )
        )

        if (
            width <= 0
            or height <= 0
        ):
            return None

        x2 = (
            x
            + width
            - 1
        )

        y2 = (
            y
            + height
            - 1
        )

        polygon = np.array(
            [
                [x, y],
                [x2, y],
                [x2, y2],
                [x, y2],
            ],
            dtype=np.int32,
        )

    return reduce_polygon_points(
        polygon,
        max_points=max_points,
    )


def instances_to_yolo_lines(
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    image_width: int,
    image_height: int,
    min_instance_area: int = 3,
    epsilon_ratio: float = 0.001,
    max_points: int = 1000,
    thin_dilate: int = 1,
) -> Tuple[
    List[str],
    Dict[str, int],
]:

    if (
        len(masks)
        != len(class_ids)
    ):
        raise ValueError(
            "masks 与 class_ids "
            "数量不一致"
        )

    lines = []

    skipped_small = 0
    skipped_invalid = 0

    for mask, class_id in zip(
        masks,
        class_ids,
    ):

        binary = as_binary_mask(
            mask
        )

        area = int(
            np.count_nonzero(
                binary
            )
        )

        if area < int(
            min_instance_area
        ):

            skipped_small += 1
            continue

        polygon = mask_to_polygon(
            binary,
            epsilon_ratio=(
                epsilon_ratio
            ),
            max_points=max_points,
            thin_dilate=thin_dilate,
        )

        if (
            polygon is None
            or len(polygon) < 3
        ):

            skipped_invalid += 1
            continue

        coordinates = []

        for x, y in polygon:

            nx = float(
                np.clip(
                    float(x)
                    / max(
                        image_width,
                        1,
                    ),
                    0.0,
                    1.0,
                )
            )

            ny = float(
                np.clip(
                    float(y)
                    / max(
                        image_height,
                        1,
                    ),
                    0.0,
                    1.0,
                )
            )

            coordinates.extend(
                [
                    f"{nx:.6f}",
                    f"{ny:.6f}",
                ]
            )

        line = (
            f"{int(class_id)} "
            + " ".join(
                coordinates
            )
        )

        lines.append(
            line
        )

    stats = {
        "input_instances": len(
            masks
        ),
        "instances_kept": len(
            lines
        ),
        "skipped_small": (
            skipped_small
        ),
        "skipped_invalid": (
            skipped_invalid
        ),
    }

    return lines, stats


# =========================================================
# 7. COORDINATE RESTORE
# =========================================================

def restore_mask_to_full_image(
    tile_mask: np.ndarray,
    x1: int,
    y1: int,
    full_height: int,
    full_width: int,
    valid_w: Optional[int] = None,
    valid_h: Optional[int] = None,
    dtype=np.uint8,
) -> np.ndarray:

    binary = as_binary_mask(
        tile_mask
    )

    if valid_w is None:
        valid_w = binary.shape[1]

    if valid_h is None:
        valid_h = binary.shape[0]

    valid_w = min(
        int(valid_w),
        binary.shape[1],
        int(full_width) - int(x1),
    )

    valid_h = min(
        int(valid_h),
        binary.shape[0],
        int(full_height) - int(y1),
    )

    full_mask = np.zeros(
        (
            int(full_height),
            int(full_width),
        ),
        dtype=dtype,
    )

    if (
        valid_w <= 0
        or valid_h <= 0
    ):
        return full_mask

    local = binary[
        :valid_h,
        :valid_w,
    ]

    if np.issubdtype(
        np.dtype(dtype),
        np.integer,
    ):

        local = (
            local > 0
        ).astype(
            dtype
        ) * np.iinfo(
            dtype
        ).max

    else:

        local = (
            local > 0
        ).astype(
            dtype
        )

    full_mask[
        y1:y1 + valid_h,
        x1:x1 + valid_w,
    ] = local

    return full_mask


# =========================================================
# 8. VISUALIZATION
# =========================================================

def class_color(
    class_id: int,
) -> Tuple[int, int, int]:

    # 固定、可重复的伪随机颜色。
    #
    # OpenCV BGR。

    class_id = int(
        class_id
    )

    rng = np.random.default_rng(
        class_id + 12345
    )

    color = rng.integers(
        60,
        230,
        size=3,
    )

    return tuple(
        int(x)
        for x in color
    )


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[
        int,
        int,
        int,
    ] = (0, 0, 255),
    alpha: float = 0.50,
) -> np.ndarray:

    if image is None:
        raise ValueError(
            "image 不能为空"
        )

    binary = (
        mask > 0
    )

    result = image.copy()

    color_layer = np.zeros_like(
        image
    )

    color_layer[:] = color

    blended = cv2.addWeighted(
        image,
        1.0 - alpha,
        color_layer,
        alpha,
        0,
    )

    result[
        binary
    ] = blended[
        binary
    ]

    return result


def create_class_overlay(
    image: np.ndarray,
    class_masks: Dict[
        int,
        np.ndarray,
    ],
    alpha: float = 0.45,
) -> np.ndarray:

    result = image.copy()

    for class_id in sorted(
        class_masks.keys()
    ):

        mask = class_masks[
            class_id
        ]

        color = class_color(
            class_id
        )

        result = create_overlay(
            result,
            mask,
            color=color,
            alpha=alpha,
        )

    return result


def draw_instance_preview(
    image: np.ndarray,
    instances: Sequence[
        Dict[str, Any]
    ],
    class_names: Optional[
        Dict[int, str]
    ] = None,
    alpha: float = 0.45,
) -> np.ndarray:

    result = image.copy()

    for index, instance in enumerate(
        instances
    ):

        mask = instance[
            "mask"
        ]

        class_id = int(
            instance.get(
                "class_id",
                0,
            )
        )

        color = class_color(
            class_id
        )

        result = create_overlay(
            result,
            mask,
            color=color,
            alpha=alpha,
        )

        features = (
            calculate_mask_features(
                mask
            )
        )

        bbox = features[
            "bbox"
        ]

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            1,
        )

        if class_names:

            class_name = (
                class_names.get(
                    class_id,
                    str(class_id),
                )
            )

        else:

            class_name = str(
                class_id
            )

        score = float(
            instance.get(
                "score",
                0.0,
            )
        )

        text = (
            f"{class_name} "
            f"{score:.2f}"
        )

        cv2.putText(
            result,
            text,
            (
                x1,
                max(
                    12,
                    y1 - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            color,
            1,
            cv2.LINE_AA,
        )

    return result


def build_instance_id_mask(
    masks: Sequence[
        np.ndarray
    ],
    dtype=np.uint16,
) -> np.ndarray:

    if not masks:
        return np.zeros(
            (0, 0),
            dtype=dtype,
        )

    height, width = (
        masks[0].shape[:2]
    )

    output = np.zeros(
        (
            height,
            width,
        ),
        dtype=dtype,
    )

    for index, mask in enumerate(
        masks,
        start=1,
    ):

        output[
            mask > 0
        ] = index

    return output
