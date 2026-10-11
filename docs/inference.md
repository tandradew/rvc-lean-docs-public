# Converting audio (inference)

Inference takes a trained model and an input recording (speech or singing, by
anyone) and produces the same recording in the model's voice: same words,
melody and timing, different timbre.

```bash
python rvc_infer.py params/infer.json --check   # validate and show the plan
python rvc_infer.py params/infer.json           # run
```

## Contents

- [A first conversion](#a-first-conversion)
- [The params file](#the-params-file)
- [What the settings do](#what-the-settings-do)
- [The index](#the-index)
- [Converting a folder](#converting-a-folder)
- [Outputs and metadata.json](#outputs-and-metadatajson)
- [Reproducibility](#reproducibility)
- [Comparing results](#comparing-results)
- [Speed](#speed)

## A first conversion

Two keys are required, the model and the input:

```json
{
  "model": "models/my-voice",
  "input": "data/test.wav"
}
```

```
$ python rvc_infer.py params/infer.json
Model:    /path/to/models/my-voice/model.pth
Index:    off (index_rate 0)
Inputs:   1 file(s)
Output:   /path/to/outputs/test_by_my-voice
Settings: f0_method=rmvpe, f0_up_key=0, protect=0.33, rms_mix_rate=0.25, sid=0, seed=3970138423
Device:   cuda:0 (NVIDIA GeForce RTX 4090), fp16
[1/1] test.wav
         -> /path/to/outputs/test_by_my-voice/test_by_my-voice.wav (10.0 s, 1.3 s to convert)
Done. Metadata: /path/to/outputs/test_by_my-voice/metadata.json
```

The input can be any common audio format, at any sample rate; the output is a
40 kHz mono wav.

## The params file

[`params/example_infer.json`](../params/example_infer.json) lists every key
with the values used in the workshop; `python rvc_infer.py -h` prints the table
below, with the defaults. Unknown keys and invalid values stop the run with a
message saying what is wrong.

| Key | Default | Meaning |
|---|---|---|
| `model` | *required* | a model folder (`models/<name>`, uses its `model.pth`) or a `.pth` file |
| `input` | *required* | an audio file, or a folder of audio files |
| `output_dir` | `"outputs"` | where the output folder is created |
| `name` | `null` | name of the output folder; `null` = `<input>_by_<model>` ([outputs](#outputs-and-metadatajson)) |
| `f0_method` | `"rmvpe"` | pitch extraction: `rmvpe`, `pm`, `harvest`, `crepe` |
| `f0_up_key` | `0` | pitch shift in semitones (12 = one octave up) |
| `protect` | `0.33` | protect consonants and breaths, 0–0.5 (0.5 = off); only matters with the index on |
| `rms_mix_rate` | `0.25` | 0–1: 0 = follow the input's loudness, 1 = keep the model's loudness |
| `index_rate` | `0.0` | 0–1: how strongly the [index](#the-index) is used; 0 (or `null`) = off |
| `index_path` | `null` | index file; `null` = `model.index` next to the model |
| `sid` | `0` | speaker id; keep 0 for models trained with this repository |
| `filter_radius` | `3` | only for `harvest`: 3 or more applies a light smoothing to the pitch curve |
| `resample_sr` | `0` | output sample rate; 0 = the model's own (40 kHz) |
| `seed` | `null` | random seed; `null` = pick one and save it in `metadata.json` |
| `overwrite` | `false` | replace an existing output folder |

Relative paths are resolved from the folder you run the command in.

> **Old params files.** Files written for the previous script
> (`infer_cli_edna.py`) are rejected with hints such as
> `"model_name" was renamed to "model"`.

## What the settings do

**`f0_up_key`: pitch shift.** The pitch (the note, or the melody of the
voice) is detected in the input and shifted by this many **semitones** (the
step between two adjacent piano keys; 12 semitones = one octave) before the
model sings or speaks it. Use it when the input and the target voice are in
different ranges: typically `12` to convert a low (male) voice to a high
(female) one, `-12` for the opposite. Shifting far outside the range the
model was trained on sounds strained.

**`f0_method`: how the pitch is detected.**

| Method | Quality | Speed | Notes |
|---|---|---|---|
| `rmvpe` | best | fast on a GPU | default; robust for singing and speech |
| `crepe` | very good | slower | |
| `pm` | lower | very fast | can miss or double notes |
| `harvest` | good | slow (CPU) | needs [pyworld](install.md#optional-pyworld); `filter_radius` ≥ 3 lightly smooths its pitch curve |

Pitch errors are the most audible failure: notes jumping an octave, or a
voice that "cracks". If you hear that, try another method.

**`rms_mix_rate`: loudness.** The model produces its own loudness contour.
`0` imposes the input's loudness (its dynamics: soft and loud passages) on the
output; `1` leaves the model's loudness as it is. The default `0.25` mostly
follows the input.

**`protect`: consonants and breaths.** Sounds without pitch (consonants like
*s*, *t*, *f*, breaths) are where conversion artefacts are most audible. In
those parts, `protect` limits how much the [index](#the-index) changes the
input: lower values protect more (0 = unpitched parts are not changed by the
index at all), and `0.5` switches the protection off. With the index off
(`index_rate: 0`), `protect` has no effect.

**`sid`: speaker id.** Models trained with this repository have one voice, at
id 0. Other ids select untrained voices from the pretrained model; keep `0`.

**`resample_sr`**: resample the output, e.g. to `44100` or `48000` for a
project at that rate. `0` keeps 40 kHz.

## The index

Each trained model comes with `model.index`: all the ContentVec feature vectors
of its training data, stored so the closest ones can be found quickly
(see [concepts.md](concepts.md)). With `index_rate` above 0, each moment of
the input's features is blended with its nearest neighbours from the training
data:

- `index_rate: 0` (default): the input's features are used as they are.
- `index_rate: 0.75`: 75 % training-data features, 25 % input features.
- `index_rate: 1`: only training-data features.

A higher rate pulls the result towards the training voice's way of
pronouncing (accent, articulation), which can make the timbre more convincing,
but it can also blur the words or add artefacts when the training data is
small or very different from the input. Try `0`, `0.5` and `0.75` and listen.
In our tests, `0.75` changed the output about twice as much as running the
same settings with a different random seed ([comparing results](#comparing-results)).

With a model folder, the index is found automatically. With a `.pth` file
(e.g. a checkpoint), set `index_path`:

```json
{
  "model": "models/my-voice/checkpoints/model_e10.pth",
  "input": "data/test.wav",
  "index_rate": 0.75,
  "index_path": "models/my-voice/model.index"
}
```

## Converting a folder

Set `input` to a folder: every audio file in it (`.wav`, `.flac`, `.mp3`,
`.ogg`, `.m4a`; not in subfolders) is converted with the same settings, and
the model is loaded only once.

```json
{"model": "models/my-voice", "input": "data/songs"}
```

## Outputs and metadata.json

Each run creates one folder in `output_dir`, named `<input>_by_<model>`. The
settings are not in the names: they are all in `metadata.json`.

```
outputs/test_by_my-voice/
├── test_by_my-voice.wav
└── metadata.json
```

- With a folder as input: `outputs/<folder>_by_<model>/`, with one
  `<file>_by_<model>.wav` per input file.
- With a checkpoint as model (`models/my-voice/checkpoints/model_e10.pth`), the
  model part of the names is `my-voice_e10`.
- **Trying several settings** on the same input and model: give each run its
  own `"name"`, e.g. `"name": "test_index075"`, and the folder is
  `outputs/test_index075/`. Without it, a second run stops before converting
  anything ("output folder already exists"); `"overwrite": true` replaces the
  earlier result instead.

`metadata.json` records everything needed to know how the file was made:

| Section | Contents |
|---|---|
| `params` | all settings, including the defaults and the `seed` actually used |
| `resolved` | the full paths of the model, index and pretrained files used |
| `model` | version, sample rate, number of speaker ids of the model |
| `device` | GPU and precision (fp16/fp32) |
| `environment` | Python, executable and package versions |
| `code` | git commit, and whether the code had uncommitted changes (`git_dirty`) |
| `outputs` | each input and output file, duration and conversion time |

## Reproducibility

The conversion involves randomness, so two runs with the same settings give
slightly different audio (not audibly, but measurably). Fixing `seed` makes a
run repeatable: with the same seed, model, settings, GPU and environment, the
output is **bit-identical**.

When `seed` is `null`, a random seed is chosen and saved in `metadata.json`.
To reproduce that file exactly, copy the seed into your params file.

This is also the fair way to compare settings: keep the seed fixed and change
one setting at a time.

## Comparing results

Listening is the final judge. To also put a number on how different two
outputs are:

```bash
python tools/compare_audio.py reference.wav other.wav [more.wav ...]
```

```
  #      len_s         sr     mel_dB    f0_corr   f0_cents    voicing     rms_dB  wave_corr
  1      10.02      40000       3.42     0.9994        4.3       94.7      +0.12     0.8156
```

| Column | Meaning |
|---|---|
| `mel_dB` | average difference between the two spectrograms, in dB: the main "how different does it sound" number. 0 = identical |
| `f0_corr` | correlation of the two pitch curves (1 = same melody) |
| `f0_cents` | typical pitch difference in cents (100 cents = 1 semitone) |
| `voicing` | % of time both agree on whether there is pitch |
| `rms_dB` | difference in overall loudness |
| `wave_corr` | sample-by-sample correlation; only meaningful for nearly identical files |

For scale: two runs with identical settings but different seeds differ by
about **1.9 dB** `mel_dB`; turning the index on (`0.75`) gave about 3.5–4 dB;
two different models about 5 dB.

## Speed

10 seconds of audio convert in about 1.3 s on an RTX 4090 (1.9 s with the
index), in about 2.5–4 s on an H100 of the workshop cluster (with the index;
reading it from the shared file system takes part of that), and in about
8–12 s on a laptop CPU. Most of the time goes to pitch
detection and synthesis; loading the models adds a few seconds per run, which
is why converting a whole folder in one run is faster than one run per file.
