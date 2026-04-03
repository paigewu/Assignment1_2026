# Copilot instructions — Assignment1_2026

Purpose
- Help an AI coding assistant be immediately productive in this repo (PyTorch-style research/training code).

Quick pointers
- Install dependencies: `pip install -r requirements.txt`.
- Common entry points:
  - Training: `TrainTools/train.py` (look at `TrainTools/train_utils.py` for config/loaders).
  - Evaluation: `EvaluateTools/evaluate.py` and `EvaluateTools/eval_utils.py`.
  - Notebook experiments: `assignment1.ipynb`.

Big-picture architecture (what to read first)
- Data layer: `Data/loader.py`, `Data/io.py`, `Data/squad.py` — these provide dataset loading, tokenization hooks and I/O formats.
- Preprocessing & utilities: `Tools/preproc.py`, `Tools/utils.py`, `Tools/download.py` — data transforms and helpers.
- Models: `Models/` contains model building blocks:
  - High-level model: `Models/qanet.py` — model assembly and top-level forward.
  - Core layers: `Models/encoder.py`, `Models/attention.py`, `Models/conv.py`, `Models/heads.py` — look here for primitive operations and where inputs/outputs are shaped.
  - Embeddings: `Models/embedding.py`.
- Layer families & conventions:
  - Activations: `Activations/` (separate files for ReLU, leaky ReLU, etc.).
  - Initializations: `Initializations/` (xavier/kaiming helpers).
  - Normalizations: `Normalizations/` (layer/group norm implementations).
- Training stack:
  - Losses: `Losses/loss.py`.
  - Optimizers: `Optimizers/` (contains custom `adam.py`, `sgd*`).
  - Schedulers: `Schedulers/` (cosine, step, lambda wrappers).
  - Training orchestration: `TrainTools/train.py`, `TrainTools/train_utils.py`.
- Evaluation: `EvaluateTools/evaluate.py` (drives evaluation using model outputs and `EvaluateTools/eval_utils.py`).

Repository conventions & patterns (concrete examples)
- Module layout: one concept per directory (e.g., `Models/` for architecture, `Tools/` for I/O & preprocessing).
- Naming: files use snake_case; model classes and layer classes live inside files named after the component (e.g., `Models/qanet.py` contains the overall model class).
- Pluggable components: activations/initializations/normalizations are modular — when modifying a layer, check these folders for matching helper functions.
- Entry and wiring: model instantiation and training loop wiring are performed in `TrainTools/train.py`. To change data flow, follow from `TrainTools/train.py` → `TrainTools/train_utils.py` → `Data/loader.py` → `Models/*`.

How to make safe, focused changes
- If modifying shapes or interfaces, update both the producing module and the consumer (common mismatch: `Models/encoder.py` outputs vs `Models/heads.py` inputs).
- When adding a new layer or initializer, place it in the matching folder (`Activations/`, `Initializations/`, or `Normalizations/`) and import it in the calling model file.
- Follow existing tests of style: most modules expose plain functions or class methods without CLI flags — prefer local function edits over changing global scripts.

Developer workflows & commands
- Install: `pip install -r requirements.txt`.
- Run training (typical):
```bash
python TrainTools/train.py
```
- Run evaluation:
```bash
python EvaluateTools/evaluate.py
```
- Open quick experiments: `jupyter notebook assignment1.ipynb`.

Integration points & external expectations
- Data formats: dataset utilities in `Data/` assume SQuAD-like structures — be mindful when changing serialization in `Data/io.py` and `Data/squad.py`.
- Evaluator: `EvaluateTools/evaluate.py` expects model outputs in the format produced by `Models/heads.py` — check `EvaluateTools/eval_utils.py` for exact keys and expected JSON schema.

What to look for when debugging
- Start at `TrainTools/train.py` to follow runtime configuration, seed setting, dataset creation and optimizer setup.
- Check `Models/qanet.py` for forward pass and `Models/encoder.py` for positional/shape handling.
- Common failure modes: shape mismatch between encoder output and head input; missing device/cuda moves (search for `.to(` or `.cuda()` usages).

Examples to reference while coding
- Add a new optimizer: copy pattern from `Optimizers/adam.py` and register usage in `TrainTools/train_utils.py`.
- Add a normalization variant: follow `Normalizations/layernorm.py` layout and replace imports in `Models/*`.

When to ask the repo owner
- If you cannot find the expected CLI flags or config parsing (look in `TrainTools/train_utils.py`) — ask for the intended configuration format.
- If model outputs don't match evaluator inputs — request an example prediction JSON produced by the original training run.

If anything above is unclear or you want a different emphasis (tests, CI, or a smaller quick-start), tell me which section to expand.
