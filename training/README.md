# Training — Jan (Model)

Fine-tune **YOLO26** to detect climbing holds (single class `hold`). This is the
**graded core**: transfer learning + evaluation.

## Inputs (from Diego)
- `data/data.yaml` (single class `hold`, `nc: 1`) + `data/data_card.md`.
- Get it by running Diego's pipeline in your environment (Colab/GPU):
  ```bash
  pip install -r requirements.txt
  export ROBOFLOW_API_KEY=...        # PowerShell: $env:ROBOFLOW_API_KEY="..."
  python data/scripts/run_pipeline.py
  ```

## Train
```bash
python training/train.py             # reads config.yaml: training.model/epochs/batch/lr, image_size=640
```
→ copies the best checkpoint to **`models/best.pt`** (the one agreed location).

## Evaluate
```bash
python training/eval.py              # mAP50, mAP50-95 (+ PR curve, confusion matrix, sample grid)
```

## The graded notebook
Lead with the **transfer-learning story**: COCO-pretrained baseline → fine-tuned on
`data.yaml` → before/after mAP. Build the Colab notebook around `train.py` + `eval.py`
so the narrative (dataset → fine-tune → eval → inference) reads top-to-bottom.

## Handoffs
- `models/best.pt` → **Ignacio**.
- metrics + figures → **Dalton**.

## Conventions (do not break)
- `imgsz = 640` (config `image_size`). Single class `hold` (config `classes`).
- Weights at `models/best.pt` (config `paths.weights`). Read everything from `config.yaml` — no hardcoded paths.
- Seed runs for reproducibility (config `training.seed`).
