# Troubleshooting

Find your problem by the message you see (search this page for a few words of
it), or by what you hear.

## Contents

- [First steps](#first-steps)
- [Reading the messages](#reading-the-messages)
- [Installation](#installation)
- [Before a run starts: params and files](#before-a-run-starts-params-and-files)
- [During training](#during-training)
- [During conversion](#during-conversion)
- [The result sounds wrong](#the-result-sounds-wrong)
- [Asking for help](#asking-for-help)

## First steps

1. **Run the environment check**: `python tools/check_env.py`. It catches most
   setup problems (packages, GPU, ffmpeg, models) and says how to fix them.
2. **Validate the params file**: `python rvc_train.py params.json --check` or
   `python rvc_infer.py params.json --check` lists every problem with the file
   and the paths in it, without running anything.
3. **Read the last error** in the output (next section).

## Reading the messages

**Problems with the params file or files** are listed together before
anything runs:

```
rvc_train: problem(s) with params/train.json:
  - unknown parameter "total_epochs" (did you mean "total_epoch"?)
  - trainset_dir not found: /home/me/rvc-lean/data/my-voce
```

**A failed training stage** stops the run with the stage name and its log:

```
rvc_train: stage "train" failed (exit code 1); see the output above and .../models/my-voice/train.log
Fix the problem and run again with "resume": true to keep the stages already done.
```

**Python tracebacks** (`Traceback (most recent call last): ...`): read from
the **bottom**. The last lines name the actual error; earlier errors in the
same traceback are often a consequence or an expected situation. For example,
when a pretrained file is broken, the traceback first shows an `IndexError` in
`latest_checkpoint_path` (the normal "no checkpoint to resume from yet" case)
and then the real error, `UnpicklingError: invalid load key`.

**Hidden warnings.** Known harmless warnings from PyTorch and other libraries
are hidden to keep the output readable. To see everything, run with
`PYTHONWARNINGS=default python rvc_train.py ...`.

## Installation

| Symptom | Cause and fix |
|---|---|
| `setup_env.sh`: `error: Python 3.x is not supported` | Use Python 3.8 (see [install.md](install.md#requirements)); 3.11+ cannot work (fairseq) |
| pip mentions **`invalid metadata`** or cannot find **`omegaconf==2.0.6`** | pip is 24.1 or newer. In the environment: `python -m pip install "pip<24.1"`, then install again |
| `DEPRECATION: omegaconf 2.0.6 has a non-standard dependency specifier` | Expected with `pip<24.1`; harmless |
| pip tries to **compile** something (`Building wheel for ...`, compiler errors) | You are not using `requirements.txt`, or not on Linux / Python 3.8. With them, everything installs from pre-built packages |
| `No matching distribution found for nvidia-...` or `triton` | Not Linux: `requirements.txt` is pinned for Linux with CUDA ([notes](install.md#notes)) |
| `check_env.py`: `[FAIL] CUDA GPU: not available` | Not on a GPU node, no NVIDIA driver, or CPU-only PyTorch installed. Check `nvidia-smi` |
| `check_env.py`: `[FAIL] CUDA: test computation failed` | The NVIDIA driver is too old for CUDA 12.1: update the driver, or ask your cluster admins |
| `check_env.py`: `[FAIL] cuDNN: 9.1.0: training on this GPU collapses` | Install cuDNN 9.5.1 in the environment: `pip install --no-deps nvidia-cudnn-cu12==9.5.1.17` ([why](#training-collapses-the-result-is-noise-or-silence)) |
| `check_env.py`: `[FAIL] ffmpeg: not found on PATH` | Load or install ffmpeg (`module load ffmpeg`, `conda install ffmpeg`, `apt install ffmpeg`) |
| `check_env.py`: `[FAIL] model ...: missing` / `wrong size` | Run `python tools/download_models.py` (delete a wrong-size file first) |
| `download_models.py`: `checksum mismatch` or `download failed` | Run it again. A failed or corrupted download is never put in place (a leftover `*.part` file is overwritten on the next run) |

## Before a run starts: params and files

These are reported by `--check` and at the start of every run. Nothing has
been computed or changed yet.

| Message (starts with) | Fix |
|---|---|
| `params file not found` | Check the path to the JSON file |
| `... is not valid JSON (line L, column C)` | A syntax error at that position: common causes are a missing or extra comma, single quotes instead of double quotes, or a comment (JSON has no comments) |
| `unknown parameter "..." (did you mean "..."?)` | A typo in a key name |
| `"..." was renamed to "..."`, `"..." is no longer a parameter` | An old params file: rename the key, or remove it |
| `missing required parameter "..."` | Add the key (`exp_name`/`trainset_dir` for training, `model`/`input` for inference) |
| `"..." must be ..., got ...` | Wrong type or range, e.g. `"save_latest_only": "No"` instead of `false`, or `"batch_size": 0` |
| `trainset_dir ... has no audio files` | The folder exists but has no `.wav/.flac/.mp3/.ogg/.m4a` files |
| `... not found ... (looked in /some/folder)` | Relative paths are resolved from the folder where you run the command, shown in the message |
| `missing weights file ...` | Run `python tools/download_models.py` |
| `ffmpeg not found on PATH (needed to read the training audio)` | Load or install ffmpeg |
| `experiment "..." already exists` | Choose: `"resume": true` to continue it, `"overwrite": true` to delete it and start again, or a new `exp_name` |
| `nothing to resume` | `resume` needs an existing experiment with that `exp_name` |
| `this experiment already finished N epochs` | To train longer with `resume`, raise `total_epoch` above N |
| `resume uses trainset_dir ..., but the experiment was started with ...` | Resuming must use the same data; use a new `exp_name` for new data |
| `no CUDA GPU found, but "f0_method" is "rmvpe_gpu"` | Run on a GPU node, or (slowly) use `"f0_method": "rmvpe"` |
| `GPU index N requested, but only M GPU(s) visible` | `gpus`/`gpus_rmvpe` count from 0 among the GPUs your job can see |
| `f0_method "harvest" needs the optional package pyworld` | `pip install pyworld==0.3.2` (needs a compiler), or use `rmvpe` |
| `output folder already exists` (inference) | `"overwrite": true`, or change `output_dir` or the settings |
| `index_rate > 0 needs an index ...` | The model has no `model.index` next to it: set `index_path`, or `"index_rate": 0` |
| `"sid" is N but this model has M speaker ids` | Use `"sid": 0` |

## During training

**`CUDA out of memory`** (`torch.cuda.OutOfMemoryError`) during the train
stage: lower `batch_size` (e.g. 32 → 16 → 8), then run again with
`"resume": true`. Also check with `nvidia-smi` that no other process is using
the GPU.

**`WARNING: N file(s) could not be read and were skipped`** (preprocess): the
listed files are damaged or not really audio; details are in
`training/preprocess.log`. The run continues without them.

**`WARNING: pitch extraction failed for N segment(s)`** or **`feature
extraction failed ...`**: those segments are left out of training; details are
in `training/extract_f0_feature.log`. A few are harmless; many point to a
problem with the data or the environment.

**`preprocess produced no usable segments`**: none of the files could be read,
or they are all silence. Check `training/preprocess.log` and that `ffmpeg`
works.

**The job was killed or hit its time limit**: run again with
`"resume": true`. Finished stages are skipped, and training continues from the
last saved checkpoint, so at most `save_epoch` epochs are lost. A smaller
`save_epoch` loses less.

**The f0 or index stage is slow**: both are partly CPU and disk bound. They
take a few minutes per ~1000 segments (see the
[timings](training.md#how-long-it-takes)), more on slow disks. Keep
experiments on fast storage if you can.

**`params.json` says `"status": "running"`** but nothing is running: the run
was interrupted (killed, Ctrl-C, node failure). Run again with `"resume": true`.

### Training collapses: the result is noise or silence

**Signs**: the converted audio is silence or incomprehensible noise, for every
input. In the training log, `loss_mel` rises instead of falling (e.g. 26 → 40
→ 60, where a healthy run goes 26 → 20 → 19) while `loss_disc` drops towards 0,
sometimes ending in `nan`. With NaN losses, training stops with
`training diverged: the losses have been NaN/inf for 50 steps in a row`.

**Known cause**: cuDNN 9.1, the version that comes with PyTorch 2.4.1,
computes wrong gradients on H100 GPUs. The model trains on wrong updates and
falls apart within a few hundred steps, in fp16 and fp32 alike. Conversion is
not affected, so a model trained elsewhere works fine on the same machine.

**Fix**: `python tools/check_env.py` shows the cuDNN version. Install 9.5.1 in
the environment (`pip install --no-deps nvidia-cudnn-cu12==9.5.1.17`, or run
`tools/setup_env.sh` again), then train again with `"overwrite": true`.

If the cuDNN version is right and training still collapses: try
`"fp16": false`, and compare the training data with that of a machine where
training works (`tools/compare_training_data.py`).

## During conversion

**`conversion failed: ... the input is silent`**: the input is silence, as
the conversion reads it (mixed to mono, 16 kHz). If the file plays normally
elsewhere, it may be stereo with channels that cancel each other when mixed:
convert one channel.

**`conversion failed: ... the output is silent`** or **`... invalid values
(NaN/inf)`**: the model is broken, most likely from a training that collapsed
([above](#training-collapses-the-result-is-noise-or-silence)). Train it again.

**`ERROR: conversion failed for <file>`** followed by a traceback: read its
last line. A common cause is an input file that is not valid audio.

**Crash without a message (`Segmentation fault`, exit code 139) when the index
is used**: faiss and PyTorch can clash when both use several CPU threads.
`rvc_infer.py` sets `OMP_NUM_THREADS=1` to prevent this; if you set
`OMP_NUM_THREADS` yourself, set it to 1.

## The result sounds wrong

Compare against changing one thing at a time, with a fixed `seed`
([reproducibility](inference.md#reproducibility)).

| What you hear | Try |
|---|---|
| Notes jump by an octave, the voice "cracks" | Pitch detection errors: another `f0_method` (`rmvpe` → `crepe`); a cleaner input (no accompaniment, little reverb) |
| Strained, unnatural, metallic in high or low passages | The pitch is outside the model's range: adjust `f0_up_key` (e.g. ±12 between male and female voices) |
| Still sounds like the input speaker | Raise `index_rate` (0.5–0.75); train more epochs; check that `sid` is 0 |
| Words blurred, wrong sounds, artefacts | Lower `index_rate`; the training data may not cover the input's language or sounds |
| Harsh or smeared consonants and breaths | Lower `protect` (e.g. 0.2), which only matters when the index is on; or lower `index_rate` |
| Loudness pumps, or does not follow the input | Adjust `rms_mix_rate`: towards 0 to follow the input's dynamics, towards 1 for the model's own |
| Only noise or silence, for every input | The model is broken: see [Training collapses](#training-collapses-the-result-is-noise-or-silence) |
| Noisy, buzzy, "underwater" at every setting | Undertrained: train longer and compare checkpoints. Or the training data is noisy: the model reproduces it |
| Good on some inputs, bad on others | The training data does not cover what the bad inputs do (range, style, singing vs speech) |

## Asking for help

Include:

1. the command you ran and the **params file**;
2. the **console output**, from the start (at least the plan and the last 30 lines);
3. `models/<exp_name>/params.json` (training) or the output's `metadata.json`
   (inference): they record the settings, environment and code version;
4. the output of `python tools/check_env.py`.
