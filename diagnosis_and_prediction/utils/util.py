import os
import csv
import random
import numpy as np

import torch
from torch.utils.data import Sampler

import pickle

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_weights_for_balanced_classes_split(dataset):
    num_classes = 4
    N = float(len(dataset))
    cls_ids = [[] for i in range(num_classes)]
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        cls_ids[label].append(idx)
    weight_per_class = [N / len(cls_ids[c]) for c in range(num_classes)]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        label = dataset.cases[idx][4]
        weight[idx] = weight_per_class[label]
    return torch.DoubleTensor(weight)


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class CV_Meter:
    def __init__(self, fold):
        self.fold = fold
        # Updated header to include all metrics
        self.header = [
            "folds", "best_epoch", 
            # Core metrics
            "Macro_AUC", "Macro_ACC", "Macro_F1", 
            "Weighted_AUC", "Weighted_ACC", "Weighted_F1",
            # Clinical cutoff metrics
            "Sens_at_90_Spec", "Sens_at_95_Spec", "Spec_at_90_Sens", "Spec_at_95_Sens",
            # PPV/NPV at clinical cutoffs
            "PPV_at_90_Spec", "NPV_at_90_Spec", "PPV_at_95_Spec", "NPV_at_95_Spec",
            "PPV_at_90_Sens", "NPV_at_90_Sens", "PPV_at_95_Sens", "NPV_at_95_Sens",
            # Clinical metrics
            "Macro_Sensitivity", "Macro_Specificity", "Macro_PPV", "Macro_NPV", "Macro_PRAUC",
            "Weighted_Sensitivity", "Weighted_Specificity", "Weighted_PPV", "Weighted_NPV", "Weighted_PRAUC",
            # Calibration
            "ECE",
        ]
        self.rows = []
        self.bootstrap_samples = {}  # Store raw bootstrap samples for statistical testing

    def updata(self, epoch, val_score, test_score=None, bootstrap_samples=None):
        # convert the tensor to float - handle both tensors and already-float values
        val_score = {k: v.item() if hasattr(v, 'item') else v for k, v in val_score.items()}
        if test_score is not None:
            test_score = {k: v.item() if hasattr(v, 'item') else v for k, v in test_score.items()}
        
        # Build row with all metrics
        row = [len(self.rows), epoch]
        
        # Core metrics
        row.extend([
            round(val_score["Macro_AUC"], 4),
            round(val_score["Macro_ACC"], 4),
            round(val_score["Macro_F1"], 4),
            round(val_score["Weighted_AUC"], 4),
            round(val_score["Weighted_ACC"], 4),
            round(val_score["Weighted_F1"], 4),
        ])
        
        # Clinical cutoff metrics
        row.extend([
            round(val_score.get("Sens_at_90_Spec", 0.0), 4),
            round(val_score.get("Sens_at_95_Spec", 0.0), 4),
            round(val_score.get("Spec_at_90_Sens", 0.0), 4),
            round(val_score.get("Spec_at_95_Sens", 0.0), 4),
        ])

        # PPV/NPV at clinical cutoffs
        row.extend([
            round(val_score.get("PPV_at_90_Spec", 0.0), 4),
            round(val_score.get("NPV_at_90_Spec", 0.0), 4),
            round(val_score.get("PPV_at_95_Spec", 0.0), 4),
            round(val_score.get("NPV_at_95_Spec", 0.0), 4),
            round(val_score.get("PPV_at_90_Sens", 0.0), 4),
            round(val_score.get("NPV_at_90_Sens", 0.0), 4),
            round(val_score.get("PPV_at_95_Sens", 0.0), 4),
            round(val_score.get("NPV_at_95_Sens", 0.0), 4),
        ])

        # Clinical metrics
        row.extend([
            round(val_score["Macro_Sensitivity"], 4),
            round(val_score["Macro_Specificity"], 4),
            round(val_score["Macro_PPV"], 4),
            round(val_score["Macro_NPV"], 4),
            round(val_score["Macro_PRAUC"], 4),
            round(val_score["Weighted_Sensitivity"], 4),
            round(val_score["Weighted_Specificity"], 4),
            round(val_score["Weighted_PPV"], 4),
            round(val_score["Weighted_NPV"], 4),
            round(val_score["Weighted_PRAUC"], 4),
        ])

        # Calibration
        row.extend([
            round(val_score.get("ECE", 0.0), 4),
        ])
        
        self.rows.append(row)
        
        if test_score is not None:
            # Test score row with bootstrap confidence intervals
            test_row = [len(self.rows), epoch]
            
            # Core metrics with confidence intervals
            test_row.extend([
                str(round(test_score["Macro_AUC_mean"], 4)) + " (" + str(round(test_score["Macro_AUC_ci_lower"], 4)) + "-" + str(round(test_score["Macro_AUC_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_ACC_mean"], 4)) + " (" + str(round(test_score["Macro_ACC_ci_lower"], 4)) + "-" + str(round(test_score["Macro_ACC_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_F1_mean"], 4)) + " (" + str(round(test_score["Macro_F1_ci_lower"], 4)) + "-" + str(round(test_score["Macro_F1_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_AUC_mean"], 4)) + " (" + str(round(test_score["Weighted_AUC_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_AUC_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_ACC_mean"], 4)) + " (" + str(round(test_score["Weighted_ACC_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_ACC_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_F1_mean"], 4)) + " (" + str(round(test_score["Weighted_F1_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_F1_ci_upper"], 4)) + ")",
            ])

            # Clinical cutoff metrics with confidence intervals
            test_row.extend([
                str(round(test_score.get("Sens_at_90_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("Sens_at_90_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("Sens_at_90_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("Sens_at_95_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("Sens_at_95_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("Sens_at_95_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("Spec_at_90_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("Spec_at_90_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("Spec_at_90_Sens_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("Spec_at_95_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("Spec_at_95_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("Spec_at_95_Sens_ci_upper", 0.0), 4)) + ")",
            ])

            # PPV/NPV at clinical cutoffs with confidence intervals
            test_row.extend([
                str(round(test_score.get("PPV_at_90_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("PPV_at_90_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("PPV_at_90_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("NPV_at_90_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("NPV_at_90_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("NPV_at_90_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("PPV_at_95_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("PPV_at_95_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("PPV_at_95_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("NPV_at_95_Spec_mean", 0.0), 4)) + " (" + str(round(test_score.get("NPV_at_95_Spec_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("NPV_at_95_Spec_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("PPV_at_90_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("PPV_at_90_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("PPV_at_90_Sens_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("NPV_at_90_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("NPV_at_90_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("NPV_at_90_Sens_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("PPV_at_95_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("PPV_at_95_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("PPV_at_95_Sens_ci_upper", 0.0), 4)) + ")",
                str(round(test_score.get("NPV_at_95_Sens_mean", 0.0), 4)) + " (" + str(round(test_score.get("NPV_at_95_Sens_ci_lower", 0.0), 4)) + "-" + str(round(test_score.get("NPV_at_95_Sens_ci_upper", 0.0), 4)) + ")",
            ])
            
            # Clinical metrics with confidence intervals
            test_row.extend([
                str(round(test_score["Macro_Sensitivity_mean"], 4)) + " (" + str(round(test_score["Macro_Sensitivity_ci_lower"], 4)) + "-" + str(round(test_score["Macro_Sensitivity_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_Specificity_mean"], 4)) + " (" + str(round(test_score["Macro_Specificity_ci_lower"], 4)) + "-" + str(round(test_score["Macro_Specificity_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_PPV_mean"], 4)) + " (" + str(round(test_score["Macro_PPV_ci_lower"], 4)) + "-" + str(round(test_score["Macro_PPV_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_NPV_mean"], 4)) + " (" + str(round(test_score["Macro_NPV_ci_lower"], 4)) + "-" + str(round(test_score["Macro_NPV_ci_upper"], 4)) + ")",
                str(round(test_score["Macro_PRAUC_mean"], 4)) + " (" + str(round(test_score["Macro_PRAUC_ci_lower"], 4)) + "-" + str(round(test_score["Macro_PRAUC_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_Sensitivity_mean"], 4)) + " (" + str(round(test_score["Weighted_Sensitivity_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_Sensitivity_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_Specificity_mean"], 4)) + " (" + str(round(test_score["Weighted_Specificity_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_Specificity_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_PPV_mean"], 4)) + " (" + str(round(test_score["Weighted_PPV_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_PPV_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_NPV_mean"], 4)) + " (" + str(round(test_score["Weighted_NPV_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_NPV_ci_upper"], 4)) + ")",
                str(round(test_score["Weighted_PRAUC_mean"], 4)) + " (" + str(round(test_score["Weighted_PRAUC_ci_lower"], 4)) + "-" + str(round(test_score["Weighted_PRAUC_ci_upper"], 4)) + ")",
            ])

            # Calibration (point estimate, not bootstrapped)
            test_row.extend([
                round(test_score.get("ECE", 0.0), 4),
            ])
            
            self.rows.append(test_row)
        
        # Store bootstrap samples if provided
        if bootstrap_samples is not None:
            fold_key = f"fold_{len(self.rows)-1}"  # Use the fold number as key
            self.bootstrap_samples[fold_key] = bootstrap_samples

    def save(self, path):
        print("save evaluation results to", path)
        
        # Save CSV results with mean and std for cross-validation
        if self.fold > 1:
            # Calculate means and stds across folds for all metrics
            means = ["mean", "mean"]  # fold and epoch columns
            stds = ["std", "std"]    # fold and epoch columns
            
            # Skip rows with confidence intervals (they contain strings)
            numeric_rows = [r for r in self.rows if all(isinstance(val, (int, float)) for val in r[2:])]
            
            # Calculate mean and std for each metric column (starting from index 2)
            for i in range(2, len(self.header)):
                if numeric_rows:  # Make sure we have numeric data
                    column_values = [r[i] for r in numeric_rows]
                    means.append(round(np.mean(column_values), 4))
                    stds.append(round(np.std(column_values), 4))
                else:
                    means.append(0.0)
                    stds.append(0.0)
            
            self.rows.append(means)
            self.rows.append(stds)
        
        # Write CSV file
        with open(path, "w", encoding="utf-8-sig", newline="") as fp:  # Changed from "a" to "w" to overwrite
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerows(self.rows)
        
        # Save bootstrap samples as pickle file
        if self.bootstrap_samples:
            bootstrap_path = path.replace('.csv', '_bootstrap_samples.pkl')
            self.save_bootstrap_samples(bootstrap_path)

    def save_bootstrap_samples(self, path):
        """Save raw bootstrap samples to pickle file for statistical testing"""
        print("save bootstrap samples to", path)
        with open(path, 'wb') as f:
            pickle.dump(self.bootstrap_samples, f)
