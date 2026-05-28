import os
import csv
import random
import numpy as np
import pickle

import torch
from torch.utils.data import Sampler


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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#* Added to save bootstrap samples for survival prediction
class Survival_Meter:
    def __init__(self):
        self.header = ["Model", "Feature", "Study", "C-Index", "CI Lower", "CI Upper", "Mean ± Std"]
        self.rows = []
        self.bootstrap_samples = {}  # Store raw bootstrap samples for statistical testing

    def update(self, args, mean_c_index, ci_lower, ci_upper, bootstrap_samples=None):
        # Estimate standard deviation from CI (95% CI = ±1.96*std)
        std_est = (ci_upper - ci_lower) / 3.92
        mean_std_format = f"{mean_c_index:.4f} ± {std_est:.4f}"

        row = [
            args.model,
            args.feature,
            args.study,
            f"{mean_c_index:.4f}",
            f"{ci_lower:.4f}",
            f"{ci_upper:.4f}",
            mean_std_format
        ]
        self.rows.append(row)
        
        # Store bootstrap samples if provided
        if bootstrap_samples is not None:
            sample_key = f"{args.model}_{args.feature}_{args.study}"
            self.bootstrap_samples[sample_key] = bootstrap_samples

    def save(self, path):
        print("save survival results to", path)
        
        # Save CSV results
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if write_header:
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