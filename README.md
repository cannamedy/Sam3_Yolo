# SAM3 → YOLO Segmentation Baseline

一套用于**图像实例分割任务**的标准基线流程。

目标是将 SAM3 自动预标注、YOLO Segmentation 数据集构建、模型训练和大图预测整理为一条结构清晰、接口统一、便于复用的流水线。

```text
Raw Images
    │
    ▼
01_sam3_prelabel.py
    │
    ▼
SAM3 Prelabels
(tile + instance masks)
    │
    ▼
02_convert_to_yolo.py
    │
    ▼
YOLO Segmentation Dataset
    │
    ▼
03_train_yolo.py
    │
    ▼
best.pt
    │
    ▼
04_predict_yolo.py
    │
    ▼
Full-image Segmentation Results
```

## 1. 适用范围

本仓库只用于：

**Image Instance Segmentation**

标准流程：

```text
SAM3
→ automatic pre-labeling
→ YOLO segmentation format
→ YOLO training
→ YOLO inference
```

不用于：

```text
YOLO Detection
Image Classification
Semantic Segmentation Pipeline
Project-specific Image Processing
```

项目特有的颜色、灰度、形态、尺寸等规则，不应直接写入本基线。

---

## 2. 核心文件

```text
01_sam3_prelabel.py
02_convert_to_yolo.py
03_train_yolo.py
04_predict_yolo.py
spot_utils.py
```

### 01_sam3_prelabel.py

使用 SAM3 对原始图片进行自动预标注。

主要流程：

```text
原图
→ 重叠切片
→ SAM3 Prompt 分割
→ mask 二值化
→ 基础实例过滤
→ 同类别实例去重
→ 保存标准实例数据
```

主要输出：

```text
data/sam3_prelabels/
├── tiles/
├── instances/
├── metadata/
├── previews/
└── dataset_meta.json
```

其中：

* `tiles/*.png`：保存模型实际处理的图像切片。
* `instances/*.npz`：保存每个 tile 的实例 mask、class_id、score、prompt_id。
* `metadata/*.json`：保存 tile 与原始大图之间的坐标关系及实例信息。
* `previews/`：用于人工快速检查 SAM3 预标注效果。
* `dataset_meta.json`：保存整个预标注数据集的基础信息和类别配置。

---

## 3. 02_convert_to_yolo.py

将 `01_sam3_prelabel.py` 输出的标准实例数据转换为 Ultralytics YOLO Segmentation 数据集。

### 设计原则

**02 不重新读取和切分原始大图。**

标准接口：

```text
01
├── tiles/*.png
├── instances/*.npz
└── metadata/*.json
        │
        ▼
02
        │
        ▼
YOLO Dataset
```

输出：

```text
data/yolo_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
└── dataset_summary.json
```

多张原图时，优先按照**原始图片级别**划分 train / val，避免同一张原图的相邻 tile 同时进入训练集和验证集。

---

## 4. 03_train_yolo.py

使用 02 生成的数据集训练 YOLO Segmentation 模型。

输入：

```text
data/yolo_dataset/data.yaml
```

输出：

```text
runs_yolo/
└── yolo_seg/
    └── weights/
        ├── best.pt
        └── last.pt
```

核心模型：

```text
best.pt
```

后续预测统一使用该模型。

---

## 5. 04_predict_yolo.py

使用训练完成的 YOLO Segmentation 模型对新图片进行预测。

流程：

```text
大图
→ overlap 切片
→ YOLO Segmentation
→ tile mask
→ 恢复原图坐标
→ overlap 合并
→ 整图 mask
```

输入图片默认放置于：

```text
data/inference_images/
```

输出：

```text
data/yolo_predictions/
└── image_name/
    ├── class_masks/
    ├── combined_mask.png
    ├── overlay.jpg
    └── summary.json
```

多类别任务会分别保存每个类别的完整 mask。

---

## 6. spot_utils.py

公共函数库。

只存放**跨项目通用能力**：

```text
文件与 JSON
图像切片
mask 基础操作
mask 几何特征
instance IoU
instance 去重
NPZ 保存与读取
mask → YOLO polygon
tile → 原图坐标恢复
基础可视化
```

### 不应加入 spot_utils.py 的内容

例如：

```text
特定灰度阈值
特定颜色阈值
圆形 / 环形规则
暗斑规则
黑色区域专用规则
某一项目固定面积阈值
某种材料特有规则
某个项目专用展示模板
```

这些逻辑应建立项目专用模块，而不是修改公共基线。

---

## 7. 推荐目录结构

```text
project/
│
├── 01_sam3_prelabel.py
├── 02_convert_to_yolo.py
├── 03_train_yolo.py
├── 04_predict_yolo.py
├── spot_utils.py
├── README.md
│
├── data/
│   ├── raw_images/
│   ├── inference_images/
│   ├── sam3_prelabels/
│   ├── yolo_dataset/
│   └── yolo_predictions/
│
└── runs_yolo/
```

---

## 8. 新项目如何使用

### Step 1：放入原始图片

将用于 SAM3 预标注的原始图片放入：

```text
data/raw_images/
```

### Step 2：修改 SAM3 类别与 Prompt

打开：

```text
01_sam3_prelabel.py
```

重点修改：

```python
CLASSES = [
    {
        "id": 0,
        "name": "target",
        "prompts": [
            "target object",
        ],
    },
]
```

单类别示例：

```python
CLASSES = [
    {
        "id": 0,
        "name": "particle",
        "prompts": [
            "small particles",
            "isolated particle objects",
        ],
    },
]
```

同一个类别可以配置多个 Prompt，用于提高召回。

多类别示例：

```python
CLASSES = [
    {
        "id": 0,
        "name": "particle",
        "prompts": [
            "small particles",
        ],
    },
    {
        "id": 1,
        "name": "crack",
        "prompts": [
            "surface cracks",
        ],
    },
]
```

---

## 9. 标准运行顺序

### 01 SAM3 自动预标注

```bash
python 01_sam3_prelabel.py
```

运行后优先检查：

```text
data/sam3_prelabels/previews/
```

确认 SAM3 预标注效果基本合理后，再继续下一步。

### 02 转换为 YOLO Segmentation 数据集

```bash
python 02_convert_to_yolo.py
```

生成：

```text
data/yolo_dataset/data.yaml
```

### 03 训练 YOLO Segmentation

```bash
python 03_train_yolo.py
```

训练完成后生成：

```text
runs_yolo/.../weights/best.pt
```

### 04 YOLO 预测

将待预测图片放入：

```text
data/inference_images/
```

运行：

```bash
python 04_predict_yolo.py
```

预测结果保存在：

```text
data/yolo_predictions/
```

---

## 10. 新项目通常需要调整的参数

原则上优先只修改各脚本顶部的：

```text
USER CONFIG
```

### 01_sam3_prelabel.py

主要调整：

```python
CLASSES

TILE_SIZE
TILE_OVERLAP

DETECTION_THRESHOLD
MASK_THRESHOLD

MIN_INSTANCE_AREA
MAX_INSTANCE_AREA

DUPLICATE_IOU_THRESHOLD
```

其中最核心的是：

```python
CLASSES
```

它决定：

* 类别数量
* 类别名称
* SAM3 Prompt

---

### 02_convert_to_yolo.py

主要调整：

```python
VAL_RATIO
NEGATIVE_RATIO

MIN_INSTANCE_AREA

POLYGON_EPSILON_RATIO
MAX_POLYGON_POINTS
THIN_OBJECT_DILATE
```

一般情况下不需要修改 02 的主体逻辑。

---

### 03_train_yolo.py

主要调整：

```python
MODEL

IMAGE_SIZE
EPOCHS
BATCH_SIZE
DEVICE

PATIENCE
```

例如：

```python
MODEL = "yolo11s-seg.pt"
```

可以根据任务规模调整为其他 YOLO segmentation 模型。

---

### 04_predict_yolo.py

主要调整：

```python
MODEL_OVERRIDE

TILE_SIZE
TILE_OVERLAP

IMAGE_SIZE

CONF_THRESHOLD
IOU_THRESHOLD

DEVICE
```

如果不指定：

```python
MODEL_OVERRIDE = None
```

脚本会自动寻找项目中的 `best.pt`。

---

## 11. 基线设计原则

### 11.1 01、02、03、04 职责分离

```text
01：只负责 SAM3 预标注
02：只负责 YOLO 数据集转换
03：只负责 YOLO 训练
04：只负责 YOLO 预测
```

不要在一个脚本中重复实现其他阶段已经完成的工作。

---

### 11.2 01 → 02 使用固定接口

01 的标准输出：

```text
tiles
instances
metadata
dataset_meta.json
```

02 直接读取这些结果。

**02 不应再次切分原始大图。**

---

### 11.3 公共逻辑只实现一次

以下功能统一放在：

```text
spot_utils.py
```

例如：

```text
图像切片
mask 处理
IoU
去重
NPZ
polygon
坐标恢复
可视化
```

01、02、04 不应分别复制实现同一功能。

---

### 11.4 项目特例不得污染基线

如果某个项目需要：

```text
灰度筛选
颜色筛选
特殊背景剔除
特殊形态过滤
目标尺寸约束
ROI 限制
材料特有规则
```

应新建项目专用模块，例如：

```text
project_filters.py
project_postprocess.py
```

而不是直接把逻辑加入：

```text
spot_utils.py
```

也不应为了单一项目修改 01—04 的标准数据接口。

---

### 11.5 优先保持接口稳定

基线的目标不是把所有项目差异自动判断出来，而是提供一套：

```text
结构稳定
职责清楚
参数集中
容易修改
容易验证
容易被 AI 理解
```

的标准实例分割流水线。

增加新功能时，应优先考虑：

```text
能否作为配置参数解决？
        ↓
不能
        ↓
能否作为独立项目模块实现？
        ↓
不能
        ↓
最后才考虑修改基线接口
```

---

## 12. 基线数据流

完整的数据关系如下：

```text
data/raw_images/
        │
        ▼
01_sam3_prelabel.py
        │
        ├── tiles/*.png
        ├── instances/*.npz
        ├── metadata/*.json
        ├── previews/*
        └── dataset_meta.json
        │
        ▼
02_convert_to_yolo.py
        │
        ├── images/train
        ├── images/val
        ├── labels/train
        ├── labels/val
        ├── data.yaml
        └── dataset_summary.json
        │
        ▼
03_train_yolo.py
        │
        ▼
runs_yolo/.../weights/best.pt
        │
        ▼
04_predict_yolo.py
        │
        ├── class_masks/
        ├── combined_mask.png
        ├── overlay.jpg
        └── summary.json
```

---

## 13. 建议开发规则

修改本仓库时建议遵守：

1. 不改变 01 → 02 的标准数据接口，除非确有必要。
2. 不将单一项目算法直接写入 `spot_utils.py`。
3. 不在 02 中重新切原图。
4. 不在 04 中依赖 SAM3。
5. 公共功能优先复用 `spot_utils.py`。
6. 新参数优先放在各脚本顶部 `USER CONFIG` 区域。
7. 修改某一阶段时，尽量不影响其他阶段。
8. 保持脚本可以按照 `01 → 02 → 03 → 04` 独立运行和检查。

---

## 14. 核心依赖

基础依赖：

```text
Python
numpy
opencv-python
Pillow
torch
ultralytics
```

01 额外依赖：

```text
SAM3
```

具体 SAM3 安装方式和模型路径根据运行环境配置。

默认通过以下参数指定：

```python
SAM3_SOURCE_PATH
SAM3_MODEL_PATH
```

---

## 15. Baseline Philosophy

这套代码首先是一套**可复用的工程基线**，而不是某一个具体图像项目的最终算法。

因此：

> Keep the baseline generic.
> Keep project-specific logic outside the baseline.
> Keep interfaces stable.
> Keep every stage independently understandable and testable.
