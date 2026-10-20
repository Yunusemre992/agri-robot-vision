# Agricultural Robot Vision

YOLO-based agricultural object detection project developed during my engineering internship, with emphasis on dataset quality, annotation consistency, training, validation, and practical inference.

## Project Goal

The goal of this project was to build an object detection pipeline for agricultural robotic applications. The work focused not only on training a detector, but also on understanding how dataset quality affects model performance.

## Workflow

```text
Dataset Collection
→ Dataset Inspection
→ Annotation / Cleaning
→ Dataset Balancing
→ YOLO Training
→ Validation
→ Inference Testing
```

## Dataset Preparation

Data was gathered from multiple public sources, including Kaggle datasets. Roboflow was used during dataset preparation and annotation workflows.

Early training experiments revealed problems such as inconsistent annotations, class imbalance, and uneven bounding-box distributions. The dataset was therefore audited, reorganized, and rebalanced before retraining.

The `scripts/` directory contains utilities used for:

- dataset auditing
- class and bounding-box inspection
- preparing data from multiple sources
- train / validation / test split organization
- dataset rebalancing
- building the final detection dataset

Large image and label folders are intentionally excluded from GitHub. The `datasets/` directory keeps metadata, manifests, source records, build reports, and audit outputs that document how the dataset was prepared.

## Model Training

The detector was trained with the Ultralytics YOLO framework using GPU resources on Kaggle.

Final model summary:

- Number of classes: 30
- Precision: ~0.855
- Recall: ~0.796
- mAP@50: ~0.866
- mAP@50-95: ~0.697

The final model checkpoint is stored as `weights/best.pt`.

## Repository Structure

```text
agri-robot-vision/
├── README.md
├── .gitignore
├── datasets/
│   ├── final/       # Build reports, manifests, audit outputs and dataset metadata
│   └── metadata/    # Class mapping and source/split metadata
├── scripts/         # Dataset preparation, auditing and rebalancing utilities
├── weights/         # Final trained checkpoint
└── results/         # Example inference outputs
```

## Tools

- Python
- Ultralytics YOLO
- PyTorch
- Roboflow
- Kaggle

## Key Engineering Lesson

The most important outcome of this project was seeing that object-detection performance depends heavily on the dataset, not only on the model architecture.

Annotation consistency, class balance, bounding-box quality, split quality, and validation methodology all had a direct effect on the final result.

## Current Status

The dataset preparation workflow, training pipeline, and final model are complete. The repository is being organized as a compact technical record of the work rather than as a full dataset mirror.
