# Evaluation Workflow: YOLOScore, FID, CLIPScore & Trainability

## 1. Install Dependencies

First, install the required packages:

```bash
pip install torch-fidelity
pip install open_clip_torch
```

For YOLOScore evaluation, install Ultralytics YOLO from source:

```bash
cd eval
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install -e .
```

---

## 2. Prepare Data for YOLOScore

Organize your data as required by Ultralytics YOLO:

```
yolo_img_path/
├── images/
│   ├── train/
│   └── val/
│       ├── 00011.jpg
│       └── ...
└── labels/
    ├── train/
    └── val/
        ├── 00011.txt
        └── ...
```

Label files (.txt) can be converted using the script: `eval/utils/xml2yolo.py`.

---

## 3. YOLOScore Training & Evaluation

Copy YOLOScore files and start training:

```bash
cd eval/ultralytics/
cp ../yoloscore/* ./
python train.py
```

- Use `train.py` to fine-tune the YOLOv8 model for YOLOScore evaluation.
- We provide a reference [checkpoint](https://pan.baidu.com/s/1YlLSGs5WIdaSJmvtoIigbg?pwd=ccdp) for DIOR-RSVG
---

## 4. Metric Calculation

Set the image path in `eval.sh`, then run:

```bash
bash eval.sh
```

This will calculate FID, CLIPScore, and YOLOScore.

---

## 5. Trainability Evaluation

We use `mmdetection==2.28.1` for trainability evaluation. Please refer to the [Step 1. Install it from source](https://mmdetection.readthedocs.io/en/stable/get_started.html) for setup.

Custom modifications for training Faster RCNN on combined synthesized and original data are provided in `eval/mmdetection`, following the same file format as the official mmdetection repo.

- Convert `metadata.jsonl` to COCO format annotation (.json) using `eval/utils/covert_jsonl_to_coco_format.py`.
- Combine synthesized and original images using `eval/utils/combine_syn_images.py`.

---

For further details, please refer to the provided scripts or