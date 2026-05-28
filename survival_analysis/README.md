# Survival Analysis

This folder contains the downstream MIL pipeline for survival prediction from pre-extracted PulmoFoundation features. It trains a discrete-time survival model, reports C-index, and saves bootstrap confidence intervals. The released examples use AttMIL with PulmoFoundation-E2 features.

For WSI preprocessing, coordinate extraction, patch cropping, and feature extraction, use [PrePATH](https://github.com/birkhoffkiki/PrePATH/tree/main). This code expects slide-level patch feature tensors that have already been extracted.

## Setup

Install dependencies from the repository root:

```bash
cd PulmoFoundation
pip install -r requirements.txt
pip install -e .
```

Run the commands below from this folder:

```bash
cd survival_analysis
```

## Public Feature Package

For reviewer testing, download the public TCGA-NSCLC PulmoFoundation-E2 feature archive from the [PulmoFoundation Hugging Face repository](https://huggingface.co/david4real/PulmoFoundation):

```text
TCGA__NSCLC.z01
TCGA__NSCLC.z02
TCGA__NSCLC.zip
```

Place all three archive parts in the same directory, then unzip from the `.zip` file:

```bash
unzip TCGA__NSCLC.zip
```

The downstream scripts use:

```bash
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files
```

The extracted `patches/` folder is included for transparency and inspection; survival commands use `pt_files/`.

## Data Format

Feature tensors should be organized as `feature_path/feature/slide.pt`:

```text
path/to/pt_files/
  PulmoFoundation-E2/
    slide_1.pt
    slide_2.pt
    slide_3.pt
```

Each `.pt` file should contain a patch-feature tensor for one slide. If a case has multiple slides, join slide filenames in the CSV with `;`.

CSV files should contain:

```text
Study,ID,Event,Status,WSI,split
TCGA-LUAD,case_id,event_time,event_status,slide_name.pt,train
TCGA-LUAD,case_id,event_time,event_status,slide_name.pt,validation
TCGA-LUAD,case_id,event_time,event_status,slide_name.pt,test
```

`Event` is the survival time, `Status` is the event indicator, and `split` should use `train`, `validation`, or `test`. The kept CSV examples are in [dataset_csv](dataset_csv/).

## Training

Update `feature_path` in [scripts/internal.sh](scripts/internal.sh), then run:

```bash
bash scripts/internal.sh
```

Set the feature root and run the desired task directly:

```bash
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files
```

Direct LUAD command:

```bash
python main.py \
  --model AttMIL \
  --csv_file ./dataset_csv/LUAD.csv \
  --feature_path ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --study LUAD \
  --modal WSI \
  --num_epoch 20 \
  --batch_size 1 \
  --lr 2e-4
```

Direct LUSC command:

```bash
python main.py \
  --model AttMIL \
  --csv_file ./dataset_csv/LUSC.csv \
  --feature_path ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --study LUSC \
  --modal WSI \
  --num_epoch 20 \
  --batch_size 1 \
  --lr 2e-4
```

Training outputs are written to `results/`.

## Evaluation

Survival examples are released as training workflows from TCGA features. Trained survival checkpoints are not bundled in this release.

To evaluate a saved survival model, train a checkpoint first, then pass `--evaluate` and set `--resume` to a results directory containing a matching checkpoint folder:

```bash
FEATURE_ROOT=/path/to/TCGA__NSCLC/pt_files

python main.py \
  --model AttMIL \
  --csv_file ./dataset_csv/LUAD.csv \
  --feature_path ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --study LUAD \
  --modal WSI \
  --evaluate \
  --resume ./results/WSI/LUAD
```

Evaluation writes a result CSV, bootstrap samples, and patient-level risk predictions into the selected checkpoint directory.
