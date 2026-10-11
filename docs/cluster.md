# Running on a cluster

How to run rvc-lean on a GPU cluster with the **SLURM** scheduler: the
workshop cluster, how to submit jobs and follow them, interactive sessions,
and what to change for another cluster.

## Contents

- [Login nodes and compute nodes](#login-nodes-and-compute-nodes)
- [The workshop cluster](#the-workshop-cluster)
- [Batch jobs](#batch-jobs)
- [Following a job](#following-a-job)
- [Interactive sessions](#interactive-sessions)
- [Time limits and resuming](#time-limits-and-resuming)
- [GPUs and CPUs](#gpus-and-cpus)
- [Storage](#storage)
- [TensorBoard](#tensorboard)
- [Another cluster](#another-cluster)

## Login nodes and compute nodes

A cluster has two kinds of machines:

- **Login nodes**: where you arrive with `ssh`. Shared by everyone, sometimes
  with internet access, **no GPUs**, and not meant for heavy computation. Use
  them to copy and edit files and to submit jobs.
- **Compute nodes**: where jobs run, with the GPUs. You do not log in to them
  directly: you ask the scheduler (SLURM) for resources, and it runs your job
  there when they are free. They often have **no internet access**.

For rvc-lean this means:

| Task | Where |
|---|---|
| copying the code and data, editing params files | login node |
| `--check` (validates a params file, no GPU needed) | login node |
| `rvc_train.py`, `rvc_infer.py`, `rvc_latent.py` | compute node with a GPU: batch job or interactive session |
| `tools/check_env.py` | both: on a login node the GPU check fails, which is expected |

## The workshop cluster

Everything is prepared in advance, because the compute nodes have no internet
access:

| What | Where |
|---|---|
| code, pretrained models (`assets/`) and datasets (`data/`), in one folder you copy to your home | `/gpfs/scratch/nct_395/Workshop_ExpandedVoices/rvc-lean` |
| Python environment, shared by everyone | `/gpfs/scratch/nct_395/Workshop_ExpandedVoices/.venv` |
| job scripts | `slurm/train.sbatch`, `slurm/infer.sbatch`, `slurm/latent.sbatch` |

Copy the folder to your home, **as `~/rvc-lean`** (the job scripts expect it
there):

```bash
cp -r /gpfs/scratch/nct_395/Workshop_ExpandedVoices/rvc-lean ~/rvc-lean
cd ~/rvc-lean
sbatch slurm/train.sbatch          # trains on the workshop dataset
```

The environment needs these modules, in every job and interactive session (the
job scripts already do this):

```bash
module load mkl
module load python/3.8.1
module load ffmpeg/7.0.1-gcc       # training only
unset PYTHONPATH
source /gpfs/scratch/nct_395/Workshop_ExpandedVoices/.venv/bin/activate
```

- `python/3.8.1` is the Python the environment was built with, and it is
  needed to run it: without the module, the environment's Python stops with
  `error while loading shared libraries: libpython3.8.so.1.0: cannot open
  shared object file`.
- `unset PYTHONPATH` is needed: the Python module points `PYTHONPATH` to its
  own packages, which would be used instead of the environment's.
- `ffmpeg/7.0.1-gcc` is only used by training, to read the audio files;
  conversion and the latent-space tool work without it. The other ffmpeg
  modules need `nvidia-hpc-sdk`, which brings NVIDIA libraries the environment
  does not need, and they are slower: decoding the training segments took
  ~13 ms per file with `ffmpeg/7.0.1-nvidia` and `ffmpeg/8.0-nvidia`, ~8.5 ms
  with `ffmpeg/7.0.1-gcc`.

### Preparing the workshop folder (instructors)

Everything lives in `/gpfs/scratch/nct_395/Workshop_ExpandedVoices`, readable
by all participants:

```
Workshop_ExpandedVoices/
  .venv/       the shared environment (tools/setup_env.sh python3 <this path>)
  rvc-lean/    git clone of the repository, plus
    assets/      the four pretrained models (tools/download_models.py)
    data/        the workshop datasets
```

The environment must have cuDNN 9.5.1: with the 9.1 that comes with PyTorch,
training on the H100 collapses ([details](troubleshooting.md#training-collapses-the-result-is-noise-or-silence)).
`setup_env.sh` installs it; for an environment created before, run
`.venv/bin/pip install --no-deps nvidia-cudnn-cu12==9.5.1.17`. Check with
`tools/check_env.py` on a GPU node: `[ ok ] cuDNN: 9.5.1`.

The models and datasets sit inside the clone, so participants copy a single
folder. `git pull` updates the code without touching them: git ignores
`data/` and the downloaded models.

After cloning and after **every `git pull`**, make the new files readable by
everyone:

```bash
chmod -R a+rX /gpfs/scratch/nct_395/Workshop_ExpandedVoices/rvc-lean
```

Test from a participant account: copy the folder, then run
`sbatch slurm/train.sbatch` and, once the model is trained,
`sbatch slurm/infer.sbatch` and `sbatch slurm/latent.sbatch`. If the code changes after participants have
copied the folder, tell them which files to copy again, e.g.
`cp /gpfs/scratch/nct_395/Workshop_ExpandedVoices/rvc-lean/rvc_infer.py ~/rvc-lean/`.
(Running `cp -r` on the whole folder again would not update it: with
`~/rvc-lean` already there, it creates `~/rvc-lean/rvc-lean`.)

## Batch jobs

There is one job script per tool, each running the example params file with
the workshop values:

| Script | Runs | Log files |
|---|---|---|
| `slurm/train.sbatch` | `rvc_train.py params/example_train.json` | `slurm-train-<jobid>.out` / `.err` |
| `slurm/infer.sbatch` | `rvc_infer.py params/example_infer.json` | `slurm-infer-<jobid>.out` / `.err` |
| `slurm/latent.sbatch` | `rvc_latent.py params/example_latent.json` | `slurm-latent-<jobid>.out` / `.err` |

Each script requests one GPU and 20 CPUs (`#SBATCH` lines), loads the modules,
activates the shared environment, moves to `~/rvc-lean` and runs the tool. The
log files are written in the folder you submitted from.

**To run your own settings**, copy an example, edit it, and change the last line
of the job script to use it:

```bash
cp params/example_train.json params/my-train.json      # edit exp_name, total_epoch, ...
# in slurm/train.sbatch:  python rvc_train.py params/my-train.json
sbatch slurm/train.sbatch
```

Validate the params file on the login node before submitting, so a typo does
not cost a place in the queue (with the modules and environment loaded as
[above](#the-workshop-cluster)):

```bash
python rvc_train.py params/my-train.json --check
```

On a login node it notes that there is no GPU there, which is fine: the job
will run on a GPU node.

## Following a job

```bash
squeue -u $USER                      # your jobs: PD = waiting, R = running
tail -f slurm-train-<jobid>.out      # the job's output as it is written
scancel <jobid>                      # stop a job
sacct -j <jobid>                     # state and exit code of a finished job
scontrol show job <jobid>            # details, including the resources it got
```

The `.out` file contains everything the tool prints: the plan, the stage
summaries, the training log lines. Errors and warnings go to the `.err` file;
look there first when a job fails. The results go to the usual places:
`models/<exp_name>/`, `outputs/` and `latent/`. `models/<exp_name>/params.json`
shows the progress of a training at any time (`"status"` and the finished
stages).

## Interactive sessions

For trying things out, an interactive shell on a GPU node is often more
convenient than batch jobs. Ask for the resources explicitly:

```bash
srun --account=nct_395 --qos=acc_training --gres=gpu:1 --cpus-per-task=20 --pty bash
# then load the modules and the environment as above, and:
cd ~/rvc-lean
python tools/check_env.py
python rvc_infer.py params/example_infer.json
```

Leave the session (`exit`) when done: the GPU stays reserved for you until
then.

## Time limits and resuming

Every job has a time limit: `#SBATCH --time`, or the default of the queue
(`--qos`) when the script does not set it, as in the workshop scripts. When it
is reached, the job is killed.

Measured on the workshop cluster (one H100, 20 CPUs), with the example params
files:

| Job | Time |
|---|---|
| training, en-alto-1 (342 files, 970 segments), 20 epochs | about 10–18 min, varying from run to run with the load on the file system: preprocess 20 s – 4 min, f0 1–5 min, features 25 s – 1.5 min, training 3.5–5.5 min (~9–11 s per epoch), index ~4 min ([details](training.md#how-long-it-takes)) |
| conversion of 10 s of audio, with the index | 2.5–4 s, plus loading the models |
| latent space, 2 layers, all 162 000 frames, `"seed": null` | about 3.5 min: extract 45 s, UMAP 2.5 min |

If a training job is killed (time limit, node failure, `scancel`), nothing is
lost but the epochs since the last checkpoint. Set `"resume": true` in the
params file and submit again: the finished stages are skipped and training
continues from the last saved checkpoint. A long training can be split into
several jobs this way, raising `total_epoch` each time
([training.md](training.md#training-longer-resuming-starting-again)).

## GPUs and CPUs

- **GPU numbering.** SLURM decides which physical GPU(s) your job gets. Inside
  the job they are numbered from 0, so `"gpus": "0"` (the default) always
  means "the first GPU of this job". rvc-lean translates this correctly even
  when SLURM sets `CUDA_VISIBLE_DEVICES`.
- **Several GPUs.** Request them (`--gres=gpu:2`) and set `"gpus": "0-1"` and
  `"gpus_rmvpe": "0-1"` to split the pitch, features and training stages over
  both. For small datasets, one GPU is enough.
- **CPUs.** Preprocessing uses `n_cpu` processes; with `"n_cpu": null` (the
  default) that is the job's `--cpus-per-task` (the plan at the top of the log
  shows it: `20 CPUs`). On the workshop cluster, `scontrol show job` reports
  twice as many CPUs (40): each core runs two hardware threads, and SLURM
  counts threads.
- **`srun` inside a job script.** On this SLURM version, `srun` does not
  inherit `--cpus-per-task` from the job. The scripts set
  `SRUN_CPUS_PER_TASK`, so a command started with `srun` still gets all 20
  CPUs.

## Storage

Clusters usually have several file systems: a small, backed-up **home**, and a
large, fast **scratch** or **project** space (often not backed up, sometimes
cleaned automatically).

A training experiment takes about **4 GB** per ~1000 segments (4.1 GB for the
workshop dataset), mostly in `models/<exp_name>/training/`, which can be
deleted when the model is final. A latent-space analysis takes about 500 MB per
layer for the workshop dataset. On the workshop cluster, each account has
80 GB, enough for many experiments. The Python environment takes about 6 GB
(on the workshop cluster it is shared, outside your home).

To keep the repository in your home but the large files on scratch, link the
folders:

```bash
mkdir -p /path/to/scratch/$USER/rvc/{models,outputs,latent}
ln -s /path/to/scratch/$USER/rvc/models models
ln -s /path/to/scratch/$USER/rvc/outputs outputs
ln -s /path/to/scratch/$USER/rvc/latent latent
```

## TensorBoard

TensorBoard runs on a node of the cluster and is viewed in your browser through
an SSH tunnel. On a login node (it is light), with the environment loaded:

```bash
tensorboard --logdir models/my-voice/training --port 6006
```

Then, on your own computer:

```bash
ssh -L 6006:localhost:6006 <user>@<login-node>
```

and open `http://localhost:6006`. If port 6006 is taken, use another number in
both commands.

## Another cluster

The job scripts contain values specific to the workshop cluster. On another
cluster, change:

| Value | In the scripts | Where to find it |
|---|---|---|
| account / project | `#SBATCH --account=nct_395` | given with your cluster access; `sacctmgr show assoc user=$USER` |
| queue | `#SBATCH --qos=acc_training` (some clusters use `--partition`) | `sinfo -s`, or the cluster documentation |
| how to request a GPU | `#SBATCH --gres=gpu:1` (some clusters: `--gpus=1`) | cluster documentation |
| time and memory | not set: the queue's defaults (add `--time`, `--mem` if needed) | cluster documentation |
| modules | `module load ...` lines | `module avail`, `module spider python` |
| environment | the `source .../.venv/bin/activate` line | where you created it |
| repository folder | `cd ~/rvc-lean` | where you cloned it |

To create the environment, on a login node with internet access, with Python
3.8 available (as a module, `unset PYTHONPATH` afterwards if it sets it):

```bash
git clone https://github.com/BSCCNS/rvc-lean.git && cd rvc-lean
tools/setup_env.sh python3.8                    # environment in .venv/, models in assets/
tools/setup_env.sh python3.8 /shared/path/.venv # or elsewhere, e.g. shared by a group
```

`setup_env.sh` ends with an environment check. On a login node the GPU check
fails (`[FAIL] CUDA GPU: not available`), which is expected: everything else
should be `ok`. See [install.md](install.md) for details. An environment can't
be moved after it is created (it contains absolute paths): create it where it
will be used. To share it with other accounts, make it readable:
`chmod -R a+rX /shared/path/.venv`.
