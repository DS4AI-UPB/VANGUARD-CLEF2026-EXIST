# `runnable/`: entry-point scripts

Thin CLI scripts that call the project's functions directly, so you don't have to navigate the code. Every flag has a sensible default (the canonical data layout), so the zero-argument form works if your data is in the standard place; override any path or hyperparameter with flags. Run `--help` on any script to see all options.

First, install the package in your environment:

```bash
pip install -e .
```

Then run the steps in order from the repository root:

```bash
python runnable/preprocess.py     # 1. OCR + cleaning  (needs Ollama: ollama pull gemma4:e4b)
python runnable/augment.py        # 2. (optional) cross-lingual augmentation
python runnable/train.py          # 3. train the multitask model
python runnable/evaluate.py       # 4. re-score the trained run on validation
```

Some overrides:

```bash
python runnable/preprocess.py --splits training
python runnable/preprocess.py --data /path/to/exist-memes

python runnable/train.py --variant ensemble
python runnable/train.py --variant single --task 2.1
python runnable/train.py --num-epochs 30 --lora-r 8

python runnable/evaluate.py --run-dir output/results/task_1/multitask --all-langs
```

Notes:
- Expected data layout under `data/exist-memes/`: `training/` and `test/`, each
  with `memes/` (images) and the raw `EXIST*.json`. `preprocess.py` creates the
  `processed_data.json` files the trainer needs.
- `train.py` covers the multitask, ensemble, and single-task variants via
  `--variant`. For the multi-seed component ablations use the module's own CLI:
  `python -m exist_2026.train.train_ablations --seed 42`.
- `evaluate.py`'s `--seed`/`--lora-*` must match what you trained with.
