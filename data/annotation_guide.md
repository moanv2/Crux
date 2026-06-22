# Annotation Guide — Climbing Holds

The detector's accuracy is capped by label quality, and hold *type* is
subjective — so **consistency matters more than being "right."** When two
people would label the same hold differently, this rubric decides. Read it
before annotating; when unsure, follow the tie-breakers, don't improvise.

## Classes (LOCKED — exact names + order)

| idx | class | what it is | grip / difficulty |
|----:|-------|------------|--------------------|
| 0 | **jug** | big, deep, positive incut you can wrap a whole hand around ("bucket") | easiest |
| 1 | **pocket** | a hole/depression that only admits 1–3 fingers; *defined by the hole*, not the outline | medium |
| 2 | **pinch** | gripped by squeezing — thumb opposing fingers; narrow/protruding with two usable faces | medium–hard |
| 3 | **sloper** | rounded, domed, no positive edge; held by open-hand friction | hard |
| 4 | **crimp** | small, thin edge; only fingertips fit, shallow | hard |
| 5 | **volume** | large geometric feature (plywood/fibreglass) bolted to the wall; holds may sit on it | structural |

## Tie-breakers (the usual arguments)

- **crimp vs jug** — depth test: only **fingertips** fit → crimp; **whole hand / multiple finger pads** and deep → jug.
- **pocket beats everything** — if a hole limits how many fingers go in, it's a **pocket**, even if otherwise juggy.
- **pinch vs jug** — is the *natural* grip a thumb-opposed squeeze? → pinch. Can you just pull down on it? → jug.
- **sloper vs jug** — any positive lip/edge to pull on → jug; pure rounded friction → sloper.
- **edge / rail** — we don't have these classes: big usable edge → **jug**, fingertip-only edge → **crimp**.
- **volume vs hold** — a volume is the big mounting *feature* (flat geometric faces, bolt/T-nut holes), not a shaped grip. A hold mounted **on** a volume gets its **own** box *and* the volume gets a box.

## Box conventions

- Box **tightly** around the hold's visible extent (don't include wall or shadow).
- **Label every hold** in the frame, including background ones — not just the "route."
- Partially occluded (e.g. a hand on it): box it if **>~50%** is visible **and** the type is clear; otherwise skip.
- One class per box. Torn between two? Apply the tie-breaker; if still stuck, pick the most likely grip and flag the image for the QA pass.

## Workflow (efficient manual typing)

1. **Fork** the GDSCMoonless set → keep a subset (~150–300 images). You inherit accurate boxes.
2. Add the 6 classes above to the project, then go box-by-box and **re-assign each `hold` → its type**. Volumes are already separated.
3. **Add your own wall photos** (draw + type, or auto-label boxes first with Grounding DINO, then correct).
4. In Roboflow: preprocess **Resize → 640** (Stretch/Fit), keep augmentation **off** here (YOLO26 augments during training — see `config.yaml augmentation:`). Generate a **Version**.
5. Put the version's `workspace / project / version` in `config.yaml`, then `python data/scripts/download.py`.

## Quality control

- One person does a final **consistency pass** over everyone's labels before the version is cut.
- Expect **class imbalance** (lots of jugs/crimps, few pockets). `stats.py` reports per-class counts; `split.py` stratifies; Coder 1 can add class weights. Don't invent pockets to balance — fix it downstream.
- `sanity_check.py` catches bad **boxes** (empty/oversized/out-of-range) but **cannot** catch a mis-typed hold — that's what this rubric and the consistency pass are for.
