"""
03_train_yolo.py

通用 Ultralytics YOLO 实例分割训练脚本。

输入：
    data/yolo_dataset/data.yaml

输出：
    runs_yolo/<RUN_NAME>/

核心输出：
    weights/best.pt

本文件不包含任何具体项目类别或图像规则。
"""

# =========================================================
# 0. USER CONFIG
# =========================================================

from pathlib import Path

PROJECT_ROOT = Path.cwd()

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "data.yaml"
)


# ---------- 模型 ----------
# 可以是：
#
# yolo11n-seg.pt
# yolo11s-seg.pt
# yolo11m-seg.pt
#
# 也可以直接改为本地绝对路径。

MODEL = "yolo11s-seg.pt"


# ---------- 输出 ----------
RUNS_ROOT = (
    PROJECT_ROOT
    / "runs_yolo"
)

RUN_NAME = "yolo_seg"


# ---------- Training ----------
IMAGE_SIZE = 512

EPOCHS = 100

BATCH_SIZE = 8

DEVICE = "0"

WORKERS = 4

PATIENCE = 30

SEED = 42


# ---------- Optimizer ----------
OPTIMIZER = "auto"

LR0 = 0.01

WEIGHT_DECAY = 0.0005


# ---------- Segmentation ----------
MASK_RATIO = 4

OVERLAP_MASK = True


# ---------- Runtime ----------
CACHE = False

EXIST_OK = False

SAVE_PERIOD = -1


# =========================================================
# 1. IMPORT
# =========================================================

import gc

from ultralytics import YOLO


# =========================================================
# 2. VALIDATION
# =========================================================

def validate_dataset():

    if not DATA_YAML.exists():

        raise FileNotFoundError(
            f"没有找到：{DATA_YAML}\n"
            "请先运行："
            "python 02_convert_to_yolo.py"
        )


def resolve_model_source():

    candidate = Path(
        MODEL
    ).expanduser()

    if candidate.is_absolute():

        if not candidate.exists():
            raise FileNotFoundError(
                f"模型不存在：{candidate}"
            )

        return str(candidate)

    local_candidate = (
        PROJECT_ROOT
        / candidate
    )

    if local_candidate.exists():
        return str(
            local_candidate.resolve()
        )

    # 如果本地不存在，则把 MODEL 原样传给 Ultralytics。
    #
    # 例如：
    # yolo11s-seg.pt
    #
    # 在有网络环境下 Ultralytics 可以自动下载。
    #
    # 离线环境请提前把权重文件放到项目目录，
    # 或把 MODEL 修改成绝对路径。

    return MODEL


def cleanup_cuda():

    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    except Exception:
        pass


# =========================================================
# 3. MAIN
# =========================================================

def main():

    validate_dataset()

    model_source = (
        resolve_model_source()
    )

    cleanup_cuda()

    print(
        "\n========== YOLO Seg Train =========="
    )

    print(
        "Dataset :",
        DATA_YAML,
    )

    print(
        "Model   :",
        model_source,
    )

    print(
        "imgsz   :",
        IMAGE_SIZE,
    )

    print(
        "epochs  :",
        EPOCHS,
    )

    print(
        "batch   :",
        BATCH_SIZE,
    )

    print(
        "device  :",
        DEVICE,
    )

    print(
        "output  :",
        RUNS_ROOT / RUN_NAME,
    )

    print(
        "====================================\n"
    )

    model = YOLO(
        model_source
    )

    try:

        results = model.train(

            data=str(
                DATA_YAML
            ),

            task="segment",

            imgsz=int(
                IMAGE_SIZE
            ),

            epochs=int(
                EPOCHS
            ),

            batch=int(
                BATCH_SIZE
            ),

            device=str(
                DEVICE
            ),

            workers=int(
                WORKERS
            ),

            project=str(
                RUNS_ROOT
            ),

            name=str(
                RUN_NAME
            ),

            exist_ok=bool(
                EXIST_OK
            ),

            seed=int(
                SEED
            ),

            deterministic=True,

            patience=int(
                PATIENCE
            ),

            optimizer=str(
                OPTIMIZER
            ),

            lr0=float(
                LR0
            ),

            weight_decay=float(
                WEIGHT_DECAY
            ),

            cache=bool(
                CACHE
            ),

            mask_ratio=int(
                MASK_RATIO
            ),

            overlap_mask=bool(
                OVERLAP_MASK
            ),

            save_period=int(
                SAVE_PERIOD
            ),

            amp=True,

            plots=True,

            val=True,

            verbose=True,
        )

    except RuntimeError as exc:

        if (
            "out of memory"
            in str(exc).lower()
        ):

            cleanup_cuda()

            raise RuntimeError(
                "CUDA 显存不足。\n"
                "建议依次尝试：\n"
                "1. 重启 Python / Jupyter 内核；\n"
                "2. 减小 BATCH_SIZE；\n"
                "3. 减小 IMAGE_SIZE。\n\n"
                f"原始错误：{exc}"
            ) from exc

        raise

    save_dir = getattr(
        results,
        "save_dir",
        None,
    )

    if save_dir is None:

        trainer = getattr(
            model,
            "trainer",
            None,
        )

        save_dir = getattr(
            trainer,
            "save_dir",
            RUNS_ROOT / RUN_NAME,
        )

    run_dir = Path(
        save_dir
    )

    best_pt = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last_pt = (
        run_dir
        / "weights"
        / "last.pt"
    )

    print(
        "\n========== Training Complete =========="
    )

    print(
        "Run dir :",
        run_dir,
    )

    print(
        "best.pt :",
        best_pt,
    )

    print(
        "last.pt :",
        last_pt,
    )

    print(
        "=======================================\n"
    )


if __name__ == "__main__":
    main()
