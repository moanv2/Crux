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
python training/eval.py              # before/after baseline + mAP50/mAP50-95 + figures
```
Vals `models/best.pt` **and** the COCO-pretrained baseline on the same `data.yaml`, then
writes `training/artifacts/metrics.json` + `metrics.md` (the handoff record for Dalton) and
copies the PR curve / confusion matrix / sample predictions there. `eval.run()` also returns
the before/after summary dict the notebook charts. The COCO baseline scores ≈0 mAP — it has
no `hold` class — which is exactly the transfer-learning gain we report.

## The graded notebook — `training/train_holds.ipynb`
Thin: imports `train.py` / `eval.py`, reads `config.yaml`, leads with the transfer-learning
story. Flow top-to-bottom: dataset → COCO baseline (≈0 mAP) → fine-tune → eval → before/after
→ inference. Run on **Colab (GPU)**.

## Tests
```bash
pytest training/test_training.py -v  # torch-free: augmentation mapping + metrics formatting
```
These import `ultralytics` lazily, so they pass with no GPU/torch (local sanity before Colab).

## Handoffs
- `models/best.pt` → **Ignacio**.
- metrics + figures → **Dalton**.

## Conventions (do not break)
- `imgsz = 640` (config `image_size`). Single class `hold` (config `classes`).
- Weights at `models/best.pt` (config `paths.weights`). Read everything from `config.yaml` — no hardcoded paths.
- Seed runs for reproducibility (config `training.seed`).
