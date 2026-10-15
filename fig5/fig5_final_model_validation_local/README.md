# Figure 5 final model validation package

This package is self-contained. It rebuilds the frozen final Figure 5 predictor from the 917 development pairs, then applies that single model to the packaged held-out, histidine, and control validation inputs. It uses paths relative to this folder only.

It does not perform feature selection, model comparison, tuning on validation data, or held-out recalibration.

Setup with venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Setup with conda:

```bash
conda env create -f environment.yml
conda activate fig5-final-validation
```

Dry run:

```bash
./run_all.sh --dry-run
```

Full run:

```bash
./run_all.sh
```

Resume after interruption:

```bash
./run_all.sh --resume
```

Main outputs are written under `outputs/`, model artifacts under `artifacts/final_model/`, and run checks under `FINAL_RUN_VALIDATION.txt`. On Apple Silicon, if normal `tensorflow` installation fails, install the TensorFlow build appropriate for your Python/macOS version, then rerun the same command.
