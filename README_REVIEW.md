# Reviewer Execution Guide

This repository is prepared for anonymous review with the model implementation
distributed as a precompiled Python wheel.

## Files reviewers need

- `dist/pchp_private_model-*.whl`
- `test.py`
- `requirements.txt`
- `config/`
- `data_list/`
- `FSSD-12/`
- `util/`
- `initmodel/`
- `exp/`
- `scripts/setup.sh`
- `scripts/run_test.sh`

The private model source files under `model/*.py` are intentionally excluded
from the public review repository.

## Run

```bash
bash scripts/setup.sh
bash scripts/run_test.sh
```

The setup script requires a Python build with SSL support because pip downloads
packages over HTTPS. A Conda environment is recommended:

```bash
conda create -n pchp_review python=3.9 openssl pip -y
conda activate pchp_review
bash scripts/setup.sh
```

By default, `run_test.sh` uses:

```text
config/SSD/fold0_resnet50_test.yaml
```

You can pass another config file:

```bash
bash scripts/run_test.sh config/SSD/fold0_resnet50_test.yaml
```

## For the author

Build the private wheel before publishing the review repository:

```bash
bash scripts/build_private_wheel.sh
bash scripts/make_review_release.sh
```

Then commit or upload the generated `review_release/` directory. Make sure
`dist/pchp_private_model-*.whl` is included.

Note: `.pyc` wheels hide source code from normal repository browsing, but they
are not cryptographic protection. For stronger protection, compile the model
with Cython/Nuitka or use a Docker/API-based review package.
