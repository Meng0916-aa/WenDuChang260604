# Environment Setup

## PyTorch is managed separately (do NOT install it via requirements.txt)

This project runs PyTorch **only** inside an existing conda environment named
`pytorch` (a CUDA build, e.g. PyTorch 2.x + CUDA). `requirements.txt`
**intentionally does not list `torch`** so that

```
pip install -r requirements.txt
```

can never overwrite or downgrade that CUDA build.

- **Do not** run `pip install torch` in the base environment.
- **Do not** add `torch` to `requirements.txt`.
- Install/verify PyTorch yourself, only when GPU training is actually needed:

```powershell
conda activate pytorch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On Windows, before running any PyTorch script, set the OpenMP workaround:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

> The current 57-track formal pipeline (metadata, conversion, conversion QC, ROI
> strategy evaluation) does **not** require PyTorch. Torch is only needed for the
> optional/legacy LSTM line.

## Base dependencies (no CUDA bindings)

The non-PyTorch dependencies install safely with pip:

```powershell
pip install -r requirements.txt
```

These are numpy, scipy, pandas, matplotlib, seaborn, pyyaml, h5py,
scikit-learn, tqdm, pytest, and (optional) openpyxl. Version ranges are kept
loose on purpose so they do not pin a single machine's patch version.

## Local machine configuration (paths)

Machine-specific absolute paths (the raw-data root) live in
`configs/local.yaml`, which is **git-ignored**. Create it from the example:

```powershell
copy configs\local.example.yaml configs\local.yaml
# then edit configs\local.yaml -> paths.raw_data_root
```

Alternatively set the environment variable (higher priority than local.yaml):

```powershell
$env:WENDUCHANG_DATA_ROOT="D:/WenDuChang-data-repo/raw_xtherm"
```

Resolution priority is: CLI `--raw-data-root` > `WENDUCHANG_DATA_ROOT` >
`configs/local.yaml` > `configs/experiments.yaml` > error. See
`docs/formal_pipeline.md` and `src/config/path_resolution.py`.

## Tests

```powershell
conda activate pytorch
$env:KMP_DUPLICATE_LIB_OK="TRUE"
python -m pytest tests -q
```

The tests do not depend on any real D-drive data path; data-dependent checks
skip when the local matrices / master CSV are absent.
