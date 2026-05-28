# Diagnosis and Prediction

This folder contains the downstream MIL pipeline for slide-level diagnosis, molecular prediction, and external validation from pre-extracted PulmoFoundation features. The released examples use ABMIL with PulmoFoundation-E2 features.

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
cd diagnosis_and_prediction
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

The extracted `patches/` folder is included for transparency and inspection; diagnosis commands use `pt_files/`.

## Data Format

Feature tensors should be organized as `root/feature/slide.pt`:

```text
path/to/pt_files/
  PulmoFoundation-E2/
    slide_1.pt
    slide_2.pt
    slide_3.pt
```

Each `.pt` file should contain a patch-feature tensor for one slide. If a case has multiple slides, join slide names in the CSV with `;`.

CSV files should contain:

```text
case,slide,label,fold
case_id,slide_name_without_pt,label_name,train
case_id,slide_name_without_pt,label_name,val
case_id,slide_name_without_pt,label_name,test
```

For fixed splits, `fold` should use `train`, `val`, and `test`. For external evaluation CSVs, set `fold` to `test`. The kept CSV examples are in [dataset_csv](dataset_csv/).

## Training

Update `ROOT_WSI` in [scripts/internal.sh](scripts/internal.sh), then run:

```bash
bash scripts/internal.sh
```

Equivalent direct command:

```bash
python main.py \
  --model ABMIL \
  --study TCGA_STK11 \
  --root ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --csv_file dataset_csv/TCGA_STK11.csv \
  --num_epoch 25 \
  --batch_size 1 \
  --lr 2e-4 \
  --tqdm
```

Training outputs are written to `results/`.

## External Evaluation

Update paths in [scripts/external.sh](scripts/external.sh), then run:

```bash
bash scripts/external.sh
```

The repository keeps ABMIL checkpoints for the released external-validation examples:

```text
results/results_42/NSCLC/[ABMIL]/
results/results_42/EGFR/[ABMIL]/
```

`scripts/external.sh` evaluates the NSCLC checkpoint by default. The EGFR block is included in the script as a commented example.

Equivalent direct NSCLC command:

```bash
python main.py \
  --model ABMIL \
  --study External_TCGA_NSCLC \
  --root ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --csv_file dataset_csv/External_TCGA_NSCLC.csv \
  --evaluate \
  --resume './results/results_42/NSCLC/[ABMIL]' \
  --tqdm
```

Equivalent direct EGFR command:

```bash
python main.py \
  --model ABMIL \
  --study External_TCGA_EGFR \
  --root ${FEATURE_ROOT} \
  --feature PulmoFoundation-E2 \
  --csv_file dataset_csv/External_TCGA_EGFR.csv \
  --evaluate \
  --resume './results/results_42/EGFR/[ABMIL]' \
  --tqdm
```

The evaluation CSV should use the same `case,slide,label,fold` schema, with `fold` set to `test`. New logs are written to `logs/`, and evaluation results/predictions are saved next to the selected checkpoint directory.
