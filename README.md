# OptFlow — Re-implementation for the CV Term Paper

Unofficial PyTorch re-implementation of **OptFlow: Fast Optimization-based Scene Flow Estimation without Supervision**
(Ahuja, Baker, Schwarting — WACV 2024, [arXiv:2401.02550](https://arxiv.org/abs/2401.02550)).

OptFlow has **no training**. For every pair of LiDAR point clouds it directly optimises

- a **flow vector for every point** (the flow field `F`), and
- one **rigid transform `T`** for the car's own motion (ego-motion),

with gradient descent (AdamW) until the first cloud, moved by `T` and `F`, lines up with the second cloud.

---

## Table of contents

1. [Project structure](#1-project-structure)
2. [One-time setup on your laptop](#2-one-time-setup-on-your-laptop)
3. [Git workflow (branches + pull requests)](#3-git-workflow-branches--pull-requests)
4. [Downloading the datasets](#4-downloading-the-datasets)
5. [Running the code — in this order](#5-running-the-code--in-this-order)
6. [Running on a Colab GPU from VS Code](#6-running-on-a-colab-gpu-from-vs-code)
7. [How the code maps to the paper](#7-how-the-code-maps-to-the-paper)
8. [Things the paper does not specify (our choices)](#8-things-the-paper-does-not-specify-our-choices)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Project structure

```
optflow/
├── configs/                   hyperparameters per dataset (YAML)
│   ├── kitti.yaml
│   ├── nuscenes.yaml          also used for Argoverse (both are sparse)
│   └── flyingthings.yaml
├── optflow/                   the method itself
│   ├── config.py              every hyperparameter + ablation on/off switches
│   ├── geometry.py            kNN search, rotation (Rodrigues), ICP
│   ├── losses.py              soft correspondence (Eq. 3-6), rigidity (Eq. 7-8)
│   ├── solver.py              the OptFlow optimisation loop (Eq. 9/10)
│   ├── metrics.py             EPE, Acc5, Acc10, outliers, angle error
│   ├── datasets.py            loads the .npz dataset files
│   └── synthetic.py           fake driving scene for testing without data
├── scripts/
│   ├── download_data.py       downloads + extracts a dataset (Windows/Mac/Linux/Colab)
│   ├── check_data.py          prints file count + keys/shapes of a dataset
│   ├── demo_synthetic.py      smoke test, no data needed
│   ├── evaluate.py            benchmark evaluation -> results .json
│   ├── diagnose.py            finds why accuracy is low (baselines, alpha sweep, lr test)
│   ├── ablation.py            reproduces Tables 3 & 4
│   └── visualize.py           red/green/blue view like Fig. 2 (needs open3d)
├── data/                      datasets go here — NOT committed to git
├── splits/                    text files listing which .npz files form the test split
├── colab_run.ipynb            notebook to run everything on a Colab GPU
├── requirements.txt
└── .gitignore                 keeps data/, *.npz, results and .venv out of git
```

---

## 2. One-time setup on your laptop

You need **Python 3.9+**, **Git** and **VS Code**.

**Step 1 — Clone the repo** (instead of downloading a zip):

```bash
git clone https://github.com/<repo-owner>/<repo-name>.git
cd <repo-name>
```

**Step 2 — Open it in VS Code:** File → Open Folder → select the repo folder. Open the terminal with `` Ctrl + ` ``.

**Step 3 — Create and activate a virtual environment:**

```powershell
python -m venv .venv
```

| Your system | Activate command |
|---|---|
| Windows PowerShell | `.venv\Scripts\activate` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Mac / Linux | `source .venv/bin/activate` |

You'll see `(.venv)` at the start of the terminal line when it's active. Activate it **every time** you open a new terminal.

> If PowerShell says *"running scripts is disabled on this system"*, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Step 4 — Install the packages:**

```powershell
pip install -r requirements.txt
```

**Step 5 — Tell VS Code to use the environment:** `Ctrl + Shift + P` → *Python: Select Interpreter* → pick the one inside `.venv`.

**Step 6 — Check everything works:**

```powershell
python scripts/demo_synthetic.py
```

It prints a *zero-flow baseline* and *OptFlow* line. OptFlow's `Acc5` should be far above the baseline's 0. On a laptop CPU this takes 20–40 seconds; that is normal.

---

## 3. Git workflow (branches + pull requests)

**Nobody pushes directly to `main`.** Each person works on their own branch and opens a Pull Request (PR).

**Before you start working, always get the latest code:**

```bash
git checkout main
git pull
```

**Make a new branch for your task** (use your name + what you're doing):

```bash
git checkout -b <yourname>/<task>        # e.g. ashik/nuscenes-loader
```

**Save and upload your work:**

```bash
git status                               # check no dataset files are listed!
git add .
git commit -m "Short description of what you changed"
git push -u origin <yourname>/<task>
```

**Open the Pull Request:** go to the repo on GitHub → click **"Compare & pull request"** → check it says `base: main ← compare: <your branch>` → create it. The other person reviews and clicks **Merge**.

**After your PR is merged**, go back to `main` and pull before starting the next task:

```bash
git checkout main
git pull
```

**Rules that avoid pain:**

- Never commit datasets or results. `.gitignore` already blocks `data/`, `*.npz`, `*.zip` and `results*.json`, but check `git status` anyway.
- Try not to edit the same file at the same time as your teammate — that causes merge conflicts.
- Commit small, focused changes with clear messages.

> First push asks for a password? GitHub doesn't accept your account password. Sign in through the VS Code GitHub prompt, or create a Personal Access Token (GitHub → Settings → Developer settings → Tokens) and use that as the password.

---

## 4. Downloading the datasets

We use the **preprocessed datasets released by the Neural Scene Flow Prior authors**
([github.com/Lilac-Lee/Neural_Scene_Flow_Prior](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior)). These are the same data the OptFlow paper compares against.

| Dataset | Size | Command | Paper test samples |
|---|---|---|---|
| KITTI | 266 MB | `python scripts/download_data.py kitti` | 50 |
| nuScenes | 73 MB | `python scripts/download_data.py nuscenes` | 310 |
| Argoverse | 370 MB | `python scripts/download_data.py argoverse` | 212 |
| FlyingThings3D | 948 MB | `python scripts/download_data.py flyingthings` | 2000 |

Start with **KITTI** (small, most important), then nuScenes and Argoverse. Leave **FlyingThings3D** for last — it's big and slow to evaluate.

### 4.1 What the download script does

1. Installs `gdown` (a Google Drive downloader) if you don't have it.
2. Downloads the archive into `data/`.
3. Extracts it into `data/<dataset-name>/`.
4. Deletes the archive and prints how many `.npz` files it found.

Run it from the **project root** (the folder that contains `scripts/`), with `(.venv)` active.

> **Windows users:** don't try `bash ...` commands in PowerShell/CMD — `bash` isn't installed there (`WSL ... execvpe(/bin/bash) failed`). Everything in this repo runs with `python`.

### 4.2 Where the data ends up

```
data/
├── kitti/          *.npz
├── nuscenes/       *.npz
├── argoverse/      *.npz
└── flyingthings/   *.npz
```

Subfolders inside these are fine — the loader searches recursively.

### 4.3 Check the download

```powershell
python scripts/check_data.py data/kitti
```

This prints the number of files and the arrays inside the first file. For KITTI you should see:

```
150 files
  gt           (33000, 3) float64      <- ground-truth flow
  pos2         (33121, 3) float64      <- point cloud at time t
  pos1         (33000, 3) float64      <- point cloud at time t-1
```

The number of points differs per file. **Run `check_data.py` on every dataset you download.** If it shows key names that aren't in the `KEYS` dictionary at the top of `optflow/datasets.py`, add them there — otherwise the loader won't find the point clouds or (worse) will silently ignore a validity mask.

**KITTI has 150 files:** the paper uses **100 for tuning** and **50 for testing**. Only report test-set numbers in the paper comparison, and never tune `k_local`/`alpha_rigid` on the test files.

Choose files with either option (works for `evaluate.py`, `ablation.py`, `diagnose.py`):

```powershell
--file_list splits/kitti_test.txt      # one file name per line (preferred)
--start 0 --end 100                    # index range into the sorted file list
```

> **Open issue:** which 50 files are the official test split is not confirmed yet (check the NSFP code or the OptFlow supplementary). Until then, say in the report which files you used.

### 4.4 If the automatic download fails

Google Drive sometimes blocks scripted downloads ("too many users have viewed or downloaded this file", quota errors). Then do it by hand:

1. Open the dataset's Google Drive link from the [NSFP README](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior#dataset) in your browser.
2. Click **Download**.
3. Extract the archive into `data/<dataset-name>/` in the project folder.
4. Run `python scripts/check_data.py data/<dataset-name>` to confirm.

### 4.5 Data on Colab vs. on your laptop

Colab is a **different computer** — it cannot see the `data/` folder on your laptop. So:

- **On your laptop:** download data only for quick CPU tests (e.g. `--limit 2`).
- **On Colab:** download again inside the notebook (`!python scripts/download_data.py kitti`). It takes seconds there.
- Colab **deletes everything** when the session ends, so you'll re-download each session (or keep a copy on Google Drive, see Section 6).

---

## 5. Running the code — in this order

| # | Step | Command |
|---|---|---|
| 1 | Smoke test (no data) | `python scripts/demo_synthetic.py` |
| 2 | Download KITTI | `python scripts/download_data.py kitti` |
| 3 | Check the files | `python scripts/check_data.py data/kitti` |
| 4 | Quick test on 5 samples | `python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048 --limit 5` |
| 5 | Full KITTI, 2048 points (Table 2) | `python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points 2048 --out results_kitti_2048.json` |
| 6 | Full KITTI, full cloud, 35 m (Table 1) | `python scripts/evaluate.py --data data/kitti --config configs/kitti.yaml --n_points full --max_range 35 --out results_kitti_full.json` |
| 7 | Diagnose + tune | `python scripts/diagnose.py --data data/kitti --config configs/kitti.yaml --start 0 --end 10`, then put the best `alpha_rigid` / `k_local` in `configs/kitti.yaml` |
| 8 | Ablation (Tables 3/4) | `python scripts/ablation.py --data data/kitti --n_points 2048 --limit 50` |
| 9 | nuScenes / Argoverse | same as step 5 with `--data data/nuscenes --config configs/nuscenes.yaml` |
| 10 | FlyingThings3D | same with `--config configs/flyingthings.yaml` (use `--limit 200` first) |

Steps 5 onwards should run on a **GPU (Colab)** — see the next section.

**Useful `evaluate.py` options:**

| Option | Meaning |
|---|---|
| `--data` | folder with the `.npz` files |
| `--config` | YAML file with hyperparameters |
| `--n_points` | `2048`, `8192`, or `full` |
| `--max_range` | drop points further than this many metres (Table 1 uses 35) |
| `--limit` | only evaluate the first N samples (good for quick tests) |
| `--file_list` | text file with the `.npz` names to use (e.g. the test split) |
| `--start`, `--end` | index range into the sorted file list |
| `--alpha_rigid`, `--k_local`, `--lr_rot`, `--lr_trans` | override one config value without editing the YAML |
| `--out` | where to save the results `.json` |

**Output metrics** (averaged over samples):

| Metric | Meaning | Better |
|---|---|---|
| `EPE` | mean end-point error, metres | lower |
| `Acc5` | % of points with error < 5 cm or < 5% (strict accuracy) | higher |
| `Acc10` | % of points with error < 10 cm or < 10% (relaxed accuracy) | higher |
| `Outliers` | % of points with error ≥ 30 cm (the paper's definition — compare this one) | lower |
| `Outliers_rel` | % with error > 30 cm **or** relative error > 10% (NSFP convention; inflated when the car barely moves) | lower |
| `AngleErr` | mean angle between predicted and true flow, radians | lower |
| `time` | seconds per sample | lower |
| `stopped_early_frac` | share of samples where early stopping triggered | — |

`results.json` also stores, per sample: file name, iterations used, whether it stopped early, and the ICP / final translation of `T`. The script prints the 5 worst files at the end.

---

## 6. Running on a Colab GPU from VS Code

1. Install the **Google Colab** extension in VS Code.
2. Open `colab_run.ipynb`.
3. Top right: **Select Kernel → Colab** → sign in → choose a **T4 GPU** runtime.
4. In the first cell, set the repo URL. To test a branch that isn't merged yet, clone it directly:

   ```python
   !git clone -b <branch-name> https://github.com/<repo-owner>/<repo-name>.git
   ```

   If the repo is **private**, use `https://<your-token>@github.com/...` — and never commit that token.

5. Run the cells top to bottom:

| Cell | What it does |
|---|---|
| 1 | Shows the GPU, clones the repo, installs requirements |
| 2 | Synthetic smoke test |
| 3 | Downloads KITTI and checks the files |
| 4 | 5-sample test, then full KITTI evaluation |
| 5 | Ablation |
| 6 | Copies the results to Google Drive |

**Colab tips:**

- **Changed code in VS Code?** Push it, then run `!git pull` in the notebook. Colab has an old copy until you do.
- **Save results to Drive** (cell 6) — the Colab disk is wiped when the session ends.
- **To avoid re-downloading data every session**, copy it to Drive once:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !cp -r data/kitti /content/drive/MyDrive/optflow_data/          # once
  !mkdir -p data && cp -r /content/drive/MyDrive/optflow_data/kitti data/   # next sessions
  ```

---

## 7. How the code maps to the paper

| Paper | Code |
|---|---|
| Eq. 2, ego-motion `T` (Sec 3.1), ICP initialisation | `solver.py` (`rot`, `trans`), `geometry.icp` |
| Eq. 3–5, local correlation weights (Sec 3.2) | `losses.soft_correspondence_loss` |
| Eq. 6, fit term, used in both directions | `solver.py` (`bidirectional`) |
| Adaptive distance threshold (Sec 3.3) | `OptFlow.threshold_at` in `solver.py` |
| Eq. 7–8, rigidity constraint (Sec 3.4) | `losses.build_rigidity_graph`, `losses.rigidity_loss` |
| Eq. 9/10, final objective | the loop in `solver.py` |
| Metrics (Sec 4) | `metrics.compute_metrics` |
| Ablations (Tables 3/4) | switches in `config.py`: `use_ego_motion`, `use_adaptive_thresh`, `use_correlation` |

**Hyperparameters from the paper:** lr = 4e-3, AdamW, 600 iterations, ε = 0.03, K_rigid = 50, threshold 2.0 m halved every 100 steps down to 0.2 m.

---

## 8. Things the paper does not specify (our choices)

Mention these in the report — they're part of the reproducibility discussion.

- **`k_local` and `alpha_rigid`** are not given → tuned on KITTI's 100 tuning samples.
- **Flow initialised to zeros** (the paper says "empty tensors").
- **Early stopping:** patience 50. The best-loss tracking restarts at every threshold stage (losses from different thresholds aren't comparable), and stopping is only allowed in the final stage. So the returned flow always comes from the strictest 0.2 m stage.
- **ICP** is run once to initialise `T`, coarse-to-fine with rejection distances 3 m → 1 m → 0.5 m (a single 1 m stage threw away correct matches when the car moved > 1 m). `T` is then refined by gradient descent.
- **Learning rates:** the paper uses one lr (4e-3) for everything. Adam moves each parameter ~lr per step, so the rotation can jump 0.004 rad/step (~16 cm at 40 m). `lr_rot` / `lr_trans` allow separate values; default `null` = paper behaviour. Test with `diagnose.py`.
- **AdamW with `weight_decay=0`** (behaves like Adam): decay would pull the flow towards zero.
- **Outliers** are reported both ways (see Section 5).
- **Loss terms are averaged**, not summed (only changes the scale of `alpha_rigid`).
- **`--max_range`** uses Euclidean distance from the sensor.
- **Table 1 is probably not reproducible with this data:** it comes from SCOOP, which uses a different KITTI preprocessing (HPLFlowNet, ~142 scenes) with a depth (z) cut at 35 m. Our main reproduction target is **Table 2**.
- **Full-cloud timings won't match the paper:** it processes clouds above 8k points in parallel chunks (supplementary), which is not implemented here.
- **Reported flow** = ego flow (`Tp − p`) + residual flow `f`, because the ground truth includes the car's own motion.

**Known behaviour:** with lr 4e-3, each step moves a point by about 4 mm while the threshold halves every 100 steps. Very large motions (more than ~0.5–0.8 m after removing ego-motion) may not be reached in time. If results look capped, try a larger `lr` or `thresh_interval` and report it as an observation.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `WSL ... execvpe(/bin/bash) failed` | You're on Windows. Use the `python scripts/...` commands, not `bash`. |
| `'Get-ChildItem' is not recognized` | You're in CMD, not PowerShell. Use `dir data\kitti` instead. |
| `Defaulting to user installation` during `pip install` | The `.venv` isn't really active. `where python` must list `.venv\Scripts\python.exe` first; if not, delete `.venv` and recreate it. |
| Runs are slow and `nvidia-smi` shows 0% GPU | CPU-only torch. `pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall` inside the venv, check `torch.cuda.is_available()`. |
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| `ModuleNotFoundError` | `(.venv)` isn't active, or VS Code uses the wrong interpreter. Activate it and reselect the interpreter. |
| `no .npz files under data/...` | Wrong `--data` path, or the download didn't extract. Run `check_data.py` on that folder. |
| `TypeError: 'NoneType' ...` in `datasets.py` | The files use key names the loader doesn't know. Run `check_data.py` and add the names to `KEYS`. |
| Google Drive download error / quota | Download manually in the browser (Section 4.4). |
| Colab runs old code | Push from VS Code, then `!git pull` in the notebook. |
| Colab lost the data / results | Colab wipes its disk each session. Re-download, and save results to Drive. |
| Very slow runs | You're on CPU. Use the Colab T4 GPU for full evaluations. |
| `git push` rejected on `main` | Push to your own branch and open a Pull Request (Section 3). |
| Dataset files show up in `git status` | Don't commit them. Check `.gitignore` is in the project root. |
