# Installation

This page sets up everything rvc-lean needs: a Python environment with the
right packages, the pretrained models, and a check that it all works.

## Requirements

| What | Why | Check with |
|---|---|---|
| **Linux** with an **NVIDIA GPU** | training needs CUDA; the pinned packages are Linux builds | `nvidia-smi` |
| NVIDIA driver for **CUDA 12.1** or newer | the PyTorch build used here is compiled for CUDA 12.1 | `nvidia-smi` (top right: "CUDA Version") |
| **Python 3.8** | fairseq 0.12.2 (which loads the ContentVec model) supports up to 3.10; everything is tested and pinned for 3.8 | `python3.8 --version` |
| **ffmpeg** | reads your training audio, whatever its format | `ffmpeg -version` |
| internet access during setup | ~3 GB of packages and ~590 MB of models are downloaded | |

> On a cluster, run the setup on a **login node** (compute nodes often have
> no internet), and training/inference on a **GPU node**. See your cluster's
> documentation for how to get Python 3.8 and ffmpeg: often `module load ...`.

Where can Python 3.8 come from? Any of these works, the setup only needs the
path to the interpreter:

- the cluster: `module load python/3.8` (name varies), then `which python3.8`
- pyenv: `pyenv install 3.8.20`, then `~/.pyenv/versions/3.8.20/bin/python`
- conda: `conda create -n py38 python=3.8`, then `<conda>/envs/py38/bin/python`

## Setup

From the repository folder:

```bash
tools/setup_env.sh python3.8              # or the full path to a Python 3.8
source .venv/bin/activate
```

`setup_env.sh <python> [venv_dir]` does six things, printing a header for each:

1. **Creates a virtual environment** in `.venv/` (or in `venv_dir` if given),
   using the Python you pass. Re-running the script reuses it.
2. **Installs `pip < 24.1`.** Newer pip versions refuse one of fairseq's
   dependencies (`omegaconf 2.0.6`). You will see a `DEPRECATION` notice about
   it during the install: that is expected and harmless.
3. **Installs `requirements.txt`**: about 3 GB, mostly PyTorch and the CUDA
   libraries. Everything comes as pre-built packages, so no compiler is needed.
4. **Replaces cuDNN 9.1 with 9.5.1.** cuDNN is NVIDIA's library for the
   convolutions on the GPU. The version that comes with PyTorch 2.4.1 computes
   wrong gradients on H100 GPUs, and training then collapses (the result is
   noise or silence). pip notes that PyTorch asks for 9.1.0: that is expected.
5. **Downloads the pretrained models** with `tools/download_models.py`.
6. **Runs `tools/check_env.py`** (below).

At the end it prints the `source .../activate` line to use the environment.
Activating is optional: running the environment's Python directly, e.g.
`.venv/bin/python rvc_train.py ...`, is equivalent.

## Checking the environment

```bash
python tools/check_env.py            # quick
python tools/check_env.py --verify   # also verifies the model files' checksums
```

Run it whenever something seems wrong, and on each new machine. A healthy
environment looks like this (shortened):

```
[ ok ] Python: 3.8.20 (/home/me/rvc-lean/.venv/bin/python)
[ ok ] package torch: 2.4.1
[ ok ] package fairseq: 0.12.2
  ...
[ ok ] package pyworld (optional): not installed; only needed for f0_method "harvest"/"dio"
[ ok ] GPU 0: NVIDIA GeForce RTX 4090, 24 GB, will use fp16
[ ok ] CUDA: torch 2.4.1+cu121, CUDA 12.1, test computation passed
[ ok ] cuDNN: 9.5.1
[ ok ] ffmpeg: ffmpeg version 4.2.7-0ubuntu0.1 ... (/usr/bin/ffmpeg)
[ ok ] model hubert/hubert_base.pt: present
[ ok ] model rmvpe/rmvpe.pt: present
[ ok ] model pretrained_v2/f0G40k.pth: present
[ ok ] model pretrained_v2/f0D40k.pth: present

28 ok, 0 warnings, 0 failed
Environment ready.
```

What the checks mean:

| Line | Checks | If it fails |
|---|---|---|
| `Python` | the interpreter version | `WARN` on 3.9/3.10 (untested), use 3.8 |
| `package ...` | installed versions against `requirements.in` | `FAIL`: run `pip install -r requirements.txt` in the environment; `WARN`: a different version is installed |
| `import ...` | only shown if a package is installed but cannot be imported | the message says why (often a missing system library) |
| `GPU`, `CUDA` | a CUDA GPU is visible and a small computation runs on it; `fp16` means the faster half-precision mode will be used | not on a GPU node, or the driver is too old for CUDA 12.1 |
| `cuDNN` | the cuDNN version is 9.5 or newer | `FAIL` on H100 (and newer) GPUs, `WARN` on others: `pip install --no-deps nvidia-cudnn-cu12==9.5.1.17` in the environment, or run `setup_env.sh` again |
| `ffmpeg` | the `ffmpeg` program is on your `PATH` | load or install ffmpeg (training cannot read audio without it) |
| `model ...` | the 4 pretrained files exist with the right size (or checksum with `--verify`) | run `python tools/download_models.py` |

The script exits with code 1 if anything failed, so it can also be used at the
start of a batch job.

## The pretrained models

rvc-lean needs four files in `assets/` (not stored in git):

| File | Size | Used for |
|---|---|---|
| `assets/hubert/hubert_base.pt` | 190 MB | turning audio into "what is being said" features (this is the ContentVec model, despite the name) |
| `assets/rmvpe/rmvpe.pt` | 181 MB | detecting the pitch (melody) of the voice |
| `assets/pretrained_v2/f0G40k.pth` | 73 MB | starting point for training the generator |
| `assets/pretrained_v2/f0D40k.pth` | 143 MB | starting point for training the discriminator |

[concepts.md](concepts.md) explains what each one does. There are two ways to
get them:

- **Download** (what `setup_env.sh` does). Skips files that already exist and
  checks every download against its SHA-256 checksum:
  ```bash
  python tools/download_models.py            # add --verify to re-check existing files
  ```
- **Link** them from another copy of RVC that already has them, to avoid
  storing them twice:
  ```bash
  tools/link_assets.sh /path/to/other/rvc/assets
  ```

## Installing without the script

The same steps by hand:

```bash
python3.8 -m venv .venv
.venv/bin/python -m pip install "pip<24.1"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps nvidia-cudnn-cu12==9.5.1.17
.venv/bin/python tools/download_models.py
.venv/bin/python tools/check_env.py
```

### Optional: pyworld

The pitch methods `harvest` and `dio` need the `pyworld` package, which is not
installed by default because it has to be compiled (needs a C compiler).
The default method, `rmvpe`, does not need it. If you want them:

```bash
pip install pyworld==0.3.2
```

## Notes

- **Other operating systems.** `requirements.txt` is pinned for Linux with
  CUDA. On macOS or Windows the CUDA packages do not exist; installing the
  packages from `requirements.in` without the pins can work for inference on
  CPU, but this is not tested and training on CPU is very slow.
- **Updating the dependencies** (maintainers): edit `requirements.in`, then
  regenerate the pinned file with the command written at the top of
  `requirements.txt` (uses [uv](https://docs.astral.sh/uv/)).
