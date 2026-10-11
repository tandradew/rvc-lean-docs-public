# Hands-on session: Expanded Voices

We will use MareNostrum to carry out the exercises. To connect from your computer, you need to use the command `ssh` in the terminal:

```bash
ssh nct01xxx@alogin1.bsc.es
```

## Accounts and credentials

We have been assigned accounts `nct01xxx`, where `xxx` are three digits, e.g. `nct01001`.

Each student will be given one account during the session. For security reasons, we will write the passwords on the board and not distribute them here.

Once you are inside MareNostrum, you can follow the instructions on the GitHub repository.

## Getting started

The first thing you need to do is to copy the workshop material into your home:

```bash
cp -r /gpfs/scratch/nct_395/Workshop_ExpandedVoices/rvc-lean ~/rvc-lean
cd ~/rvc-lean
```

This will copy the necessary code, data, and pretrained models. It is a large folder (~1 GB) and copying it can take a few minutes.

## Data

We will use the open dataset [GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger).

It contains around 80 hours of professionally recorded singing in 9 different languages. All audios are phonetically and musically annotated. The audios are in `.wav` format, and the phonetic annotations in `.TextGrid` format, the standard in phonetic studies (used by the software Praat).

Here we will only use a small subset, corresponding to the female alto singer in English, comprising around 50 minutes (`data/en-alto-1`).

## Code

The code is purely Python based (Python 3.8), building on the open source model [Retrieval-based Voice Conversion (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

The dependencies include the standard scientific computation libraries such as numpy, pandas and PyTorch, and audio processing libraries such as ffmpeg and librosa.

The virtual environment is already set up for everyone to use in the shared folder

```
/gpfs/scratch/nct_395/Workshop_ExpandedVoices/.venv
```

which you will source in the SLURM submission scripts (this is already prepared).

## Modes of use

In this workshop we will learn to do three things:

- **Train** a model given a target voice (here the female alto singer in English from GTSinger). This is voice cloning!
- Perform **inference** with the trained model, using the two audio samples provided in the workshop material: `data/voice_samples/JA-Soprano-voice_sample.wav` and `data/voice_samples/IT-Soprano-voice_sample.wav`.
- Extract the **latent space** of phonetic features from the training dataset, compress it to two dimensions with UMAP, and do some simple visualizations.

These three actions are implemented by independent Python scripts, which read their own parameter files and can be submitted to the cluster with their own SLURM scripts.

|  | Training | Inference | Latent space |
| :---- | :---- | :---- | :---- |
| Python script | `rvc_train.py` | `rvc_infer.py` | `rvc_latent.py` |
| Parameter file | `params/example_train.json` | `params/example_infer.json` | `params/example_latent.json` |
| Submission script | `slurm/train.sbatch` | `slurm/infer.sbatch` | `slurm/latent.sbatch` |
| Results in | `models/en-alto-1-20-epochs/` | `outputs/` | `latent/en-alto-1-latent/` |
| More details | [training.md](training.md) | [inference.md](inference.md) | [latent.md](latent.md) |

To submit a job, use `sbatch` with the corresponding submission script, e.g.

```bash
sbatch slurm/train.sbatch
```

Training takes about 10–20 minutes (longer when everyone trains at the same time). You don't need to wait for it: the latent space job doesn't need the trained model, so you can submit it in the meantime.

The submission scripts call the corresponding Python script with its parameter file. To change the parameters of a task, edit its parameter file.

After submission, you can monitor the progress of your job with `squeue`:

```bash
squeue --me
```

The log of each job is written to `slurm-<tool>-<jobid>.out` (and errors to `.err`) in the `rvc-lean` folder.

## Downloading data

To listen to and look at your results, you need to download the outputs to your local machine. You can do this with the command `scp`, run from a terminal on your local machine (not on MareNostrum), e.g.

```bash
scp "nct01xxx@alogin1.bsc.es:~/rvc-lean/outputs/*.wav" .
scp "nct01xxx@alogin1.bsc.es:~/rvc-lean/latent/en-alto-1-latent/umap_layer*" .
```

The second command downloads only the UMAP files (`umap_layer10.parquet`, `umap_layer12.parquet` and their `.png` plots), which is all you need for the visualization. The rest of the latent space folder is large and slow to download.

We also provide the output files in the workshop's [Google Drive](https://drive.google.com/drive/folders/1lf3E78zJuVuYufDjE0RIix571o342Ffa).

## Output audios

You can listen to the converted audio files on your local machine with any audio player. For further visualization and analysis of the waveform, you can use the free open source software [Audacity](https://www.audacityteam.org/).

## Latent space visualization

For latent space visualization, you need to download some of the output of `rvc_latent.py` to your local machine and use any visualization tool (e.g. a Jupyter notebook). In particular, you will need the UMAP compressed files

```
latent/en-alto-1-latent/umap_layer10.parquet
latent/en-alto-1-latent/umap_layer12.parquet
```

We provide a simple example notebook in `notebooks/explore_latent_workshop.ipynb`.

Note that UMAP uses a random seed which we are keeping unfixed. That means each compression will be slightly different from the rest.

## Exercises

1. Carry out training, inference and latent space extraction using the appropriate scripts. Note that we provide two sample audios you can use for inference.
2. Listen to some training audios, the original audios and the transformed ones. Can you hear the differences and similarities?
3. Change the pitch of the audio output at inference.
4. Compare the inference of two audios using/not using the index. Can you hear the difference?
5. Train for 40 epochs instead of 20 and do inference with the new model. Can you hear the difference? Monitor the various loss functions.\*
6. Can you identify consonant sounds other than /S/?
7. Repeat the analysis of the latent space in another language. You need to download the data from GTSinger.\*\*
