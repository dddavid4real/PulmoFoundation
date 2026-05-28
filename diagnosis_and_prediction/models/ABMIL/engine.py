import os
import glob
import re
from datetime import datetime
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from typing import Dict
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.bootstrapping import BootStrapper
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import AUROC
from torchmetrics.classification import (
    MulticlassRecall, MulticlassSpecificity, MulticlassPrecision, 
    MulticlassStatScores, AveragePrecision, MulticlassCalibrationError
)
from sklearn.metrics import roc_curve

import torch
import torch.nn.functional as F

import pickle

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.num_folds > 1:
            self.results_dir = os.path.join(results_dir, "fold_" + str(fold))
        else:
            self.results_dir = results_dir
        # tensorboard
        # if args.log_data:
        #     self.writer = SummaryWriter(self.results_dir, flush_secs=15)
        self.val_scores = None
        self.test_scores = None if self.args.num_folds > 1 else dict()
        self.test_scores_bootstrapped = None if self.args.num_folds > 1 else dict()
        self.test_bootstrap_samples = None
        self.filename_best = None
        self.best_epoch = 0
        self.early_stop = 0
        self.epoch = 0

    def find_latest_checkpoint(self, root_dir, model_name):
        """
        Find the latest checkpoint directory for a given model name.
        Directory format: [model_name]-[YYYY-MM-DD]-[HH-MM-SS]
        
        Args:
            root_dir: Root directory containing checkpoint folders
            model_name: Model name (self.args.feature)
        
        Returns:
            Path to the checkpoint file, or None if not found
        """
        if not os.path.isdir(root_dir):
            return None
        
        # Find all directories that match the pattern [model_name]-[date]-[time]
        matching_dirs = []
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item.startswith(f"[{model_name}]-"):
                matching_dirs.append(item)
        
        if not matching_dirs:
            return None
        
        # Sort directories by date and time to find the most recent
        def parse_dir_datetime(dir_name):
            try:
                # Parse format: [model_name]-[YYYY-MM-DD]-[HH-MM-SS]
                pattern = r'\[([^\]]+)\]-\[([^\]]+)\]-\[([^\]]+)\]'
                match = re.match(pattern, dir_name)
                
                if match:
                    model_part, date_part, time_part = match.groups()
                    # Convert time format from HH-MM-SS to HH:MM:SS
                    time_formatted = time_part.replace('-', ':')
                    datetime_str = f"{date_part} {time_formatted}"
                    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                else:
                    return datetime(1900, 1, 1)  # If parsing fails, use early date
            except:
                return datetime(1900, 1, 1)  # If parsing fails, use early date
        
        # Sort by datetime (most recent first)
        matching_dirs.sort(key=parse_dir_datetime, reverse=True)
        latest_dir = matching_dirs[0]
        latest_dir_path = os.path.join(root_dir, latest_dir)
        
        # Find checkpoint files in the directory
        try:
            all_files = os.listdir(latest_dir_path)
        except:
            return None
        
        # Filter files by extension (avoid glob due to square bracket issues)
        pth_tar_files = []
        model_best_files = []
        pth_files = []
        pt_files = []
        
        for file in all_files:
            file_path = os.path.join(latest_dir_path, file)
            if file.endswith('.pth.tar'):
                pth_tar_files.append(file_path)
                if file.startswith('model_best_'):
                    model_best_files.append(file_path)
            elif file.endswith('.pth'):
                pth_files.append(file_path)
            elif file.endswith('.pt'):
                pt_files.append(file_path)
        
        # Prefer model_best files, then any pth.tar files, then others
        if model_best_files:
            return model_best_files[0]
        elif pth_tar_files:
            return pth_tar_files[0]
        elif pth_files:
            return pth_files[0]
        elif pt_files:
            return pt_files[0]
        else:
            return None

    def compute_clinical_cutoffs(self, probs, labels):
        """
        Compute clinically meaningful metrics:
        - Sensitivity @ 90% and 95% Specificity
        - Specificity @ 90% and 95% Sensitivity
        - Per-class Sensitivity and Specificity
        
        Args:
            probs: (N, C) numpy array of softmax outputs
            labels: (N,) numpy array of true class indices
        
        Returns:
            Dictionary with clinical metrics
        """
        num_classes = probs.shape[1]
        
        # Storage for per-class metrics
        per_class_sens = []
        per_class_spec = []
        sens_at_90_spec = []
        sens_at_95_spec = []
        spec_at_90_sens = []
        spec_at_95_sens = []
        # PPV/NPV at each operating point
        ppv_at_90_spec = []
        npv_at_90_spec = []
        ppv_at_95_spec = []
        npv_at_95_spec = []
        ppv_at_90_sens = []
        npv_at_90_sens = []
        ppv_at_95_sens = []
        npv_at_95_sens = []
        
        for c in range(num_classes):
            # Binarize: class c vs rest
            y_true = (labels == c).astype(int)
            y_score = probs[:, c]
            
            # Skip if class has no positive or no negative samples
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                per_class_sens.append(np.nan)
                per_class_spec.append(np.nan)
                sens_at_90_spec.append(np.nan)
                sens_at_95_spec.append(np.nan)
                spec_at_90_sens.append(np.nan)
                spec_at_95_sens.append(np.nan)
                ppv_at_90_spec.append(np.nan)
                npv_at_90_spec.append(np.nan)
                ppv_at_95_spec.append(np.nan)
                npv_at_95_spec.append(np.nan)
                ppv_at_90_sens.append(np.nan)
                npv_at_90_sens.append(np.nan)
                ppv_at_95_sens.append(np.nan)
                npv_at_95_sens.append(np.nan)
                continue
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            specificity = 1 - fpr  # Specificity = 1 - FPR
            sensitivity = tpr      # Sensitivity = TPR
            
            # Per-class Sensitivity/Specificity at optimal threshold (Youden's J)
            youden_j = sensitivity + specificity - 1
            optimal_idx = np.argmax(youden_j)
            per_class_sens.append(sensitivity[optimal_idx])
            per_class_spec.append(specificity[optimal_idx])
            
            # Sensitivity @ 90% Specificity
            valid_idx = np.where(specificity >= 0.90)[0]
            if len(valid_idx) > 0:
                # Among valid indices, pick the one with highest sensitivity
                best_idx = valid_idx[np.argmax(sensitivity[valid_idx])]
                sens_at_90_spec.append(sensitivity[best_idx])
                # PPV/NPV at this threshold
                thresh = thresholds[best_idx]
                y_pred = (y_score >= thresh).astype(int)
                tp_t = np.sum((y_pred == 1) & (y_true == 1))
                fp_t = np.sum((y_pred == 1) & (y_true == 0))
                tn_t = np.sum((y_pred == 0) & (y_true == 0))
                fn_t = np.sum((y_pred == 0) & (y_true == 1))
                ppv_at_90_spec.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else np.nan)
                npv_at_90_spec.append(tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else np.nan)
            else:
                sens_at_90_spec.append(0.0)
                ppv_at_90_spec.append(np.nan)
                npv_at_90_spec.append(np.nan)
            
            # Sensitivity @ 95% Specificity
            valid_idx = np.where(specificity >= 0.95)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(sensitivity[valid_idx])]
                sens_at_95_spec.append(sensitivity[best_idx])
                # PPV/NPV at this threshold
                thresh = thresholds[best_idx]
                y_pred = (y_score >= thresh).astype(int)
                tp_t = np.sum((y_pred == 1) & (y_true == 1))
                fp_t = np.sum((y_pred == 1) & (y_true == 0))
                tn_t = np.sum((y_pred == 0) & (y_true == 0))
                fn_t = np.sum((y_pred == 0) & (y_true == 1))
                ppv_at_95_spec.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else np.nan)
                npv_at_95_spec.append(tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else np.nan)
            else:
                sens_at_95_spec.append(0.0)
                ppv_at_95_spec.append(np.nan)
                npv_at_95_spec.append(np.nan)
            
            # Specificity @ 90% Sensitivity
            valid_idx = np.where(sensitivity >= 0.90)[0]
            if len(valid_idx) > 0:
                # Among valid indices, pick the one with highest specificity
                best_idx = valid_idx[np.argmax(specificity[valid_idx])]
                spec_at_90_sens.append(specificity[best_idx])
                # PPV/NPV at this threshold
                thresh = thresholds[best_idx]
                y_pred = (y_score >= thresh).astype(int)
                tp_t = np.sum((y_pred == 1) & (y_true == 1))
                fp_t = np.sum((y_pred == 1) & (y_true == 0))
                tn_t = np.sum((y_pred == 0) & (y_true == 0))
                fn_t = np.sum((y_pred == 0) & (y_true == 1))
                ppv_at_90_sens.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else np.nan)
                npv_at_90_sens.append(tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else np.nan)
            else:
                spec_at_90_sens.append(0.0)
                ppv_at_90_sens.append(np.nan)
                npv_at_90_sens.append(np.nan)
            
            # Specificity @ 95% Sensitivity
            valid_idx = np.where(sensitivity >= 0.95)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(specificity[valid_idx])]
                spec_at_95_sens.append(specificity[best_idx])
                # PPV/NPV at this threshold
                thresh = thresholds[best_idx]
                y_pred = (y_score >= thresh).astype(int)
                tp_t = np.sum((y_pred == 1) & (y_true == 1))
                fp_t = np.sum((y_pred == 1) & (y_true == 0))
                tn_t = np.sum((y_pred == 0) & (y_true == 0))
                fn_t = np.sum((y_pred == 0) & (y_true == 1))
                ppv_at_95_sens.append(tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else np.nan)
                npv_at_95_sens.append(tn_t / (tn_t + fn_t) if (tn_t + fn_t) > 0 else np.nan)
            else:
                spec_at_95_sens.append(0.0)
                ppv_at_95_sens.append(np.nan)
                npv_at_95_sens.append(np.nan)
        
        # Compute macro averages (ignoring NaN values)
        results = {
            # Per-class metrics
            "Per_Class_Sensitivity": per_class_sens,
            "Per_Class_Specificity": per_class_spec,
            # Clinical cutoff metrics (macro-averaged)
            "Sens_at_90_Spec": np.nanmean(sens_at_90_spec),
            "Sens_at_95_Spec": np.nanmean(sens_at_95_spec),
            "Spec_at_90_Sens": np.nanmean(spec_at_90_sens),
            "Spec_at_95_Sens": np.nanmean(spec_at_95_sens),
            # PPV/NPV at clinical cutoffs (macro-averaged)
            "PPV_at_90_Spec": np.nanmean(ppv_at_90_spec),
            "NPV_at_90_Spec": np.nanmean(npv_at_90_spec),
            "PPV_at_95_Spec": np.nanmean(ppv_at_95_spec),
            "NPV_at_95_Spec": np.nanmean(npv_at_95_spec),
            "PPV_at_90_Sens": np.nanmean(ppv_at_90_sens),
            "NPV_at_90_Sens": np.nanmean(npv_at_90_sens),
            "PPV_at_95_Sens": np.nanmean(ppv_at_95_sens),
            "NPV_at_95_Sens": np.nanmean(npv_at_95_sens),
        }
        
        return results

    def learning(self, model, loaders, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        
        if self.args.resume is not None:
            # New logic: find checkpoint based on root directory and model name
            checkpoint_path = self.find_latest_checkpoint(self.args.resume, self.args.feature)
            
            if checkpoint_path and os.path.isfile(checkpoint_path):
                print("=> loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.val_scores = checkpoint["val_scores"]
                self.best_epoch = checkpoint["best_epoch"]
                if "test_scores" in checkpoint:
                    self.test_scores = checkpoint["test_scores"]
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint (val score: {})".format(checkpoint["val_scores"]["Macro_AUC"]))
                if self.test_scores is not None:
                    if "Macro_AUC_mean" in self.test_scores:
                        print("=> loaded checkpoint (test score: {})".format(self.test_scores["Macro_AUC_mean"]))
                    elif "Macro_AUC" in self.test_scores:
                        print("=> loaded checkpoint (test score: {})".format(self.test_scores["Macro_AUC"]))
                    else:
                        print("=> loaded checkpoint (test score keys: {})".format(list(self.test_scores.keys())))
            else:
                print("=> no checkpoint found for model '{}' in directory '{}'".format(self.args.feature, self.args.resume))

        if self.args.evaluate:
            loader = loaders[-1]
            self.test_scores, self.test_scores_bootstrapped, self.test_scores_bootstrap_samples, self.predictions = self.validate(
                loader, model, criterion, status="test"
            )
            return self.test_scores, self.test_scores_bootstrapped, self.test_scores_bootstrap_samples, self.predictions

        for epoch in range(self.best_epoch, self.args.num_epoch):
            self.epoch = epoch
            if self.args.num_folds > 1:
                train_loader, val_loader = loaders
            else:
                train_loader, val_loader, test_loader = loaders
            # train
            train_scores = self.train(train_loader, model, criterion, optimizer)
            # evaluate
            val_scores = self.validate(val_loader, model, criterion, status="val")
            is_best = (val_scores["Macro_AUC"] > self.val_scores["Macro_AUC"]) if self.val_scores is not None else True
            if self.args.num_folds > 1:
                if is_best:
                    self.val_scores = val_scores
                    self.best_epoch = self.epoch
                    self.save_checkpoint(
                        {
                            "best_epoch": self.best_epoch,
                            "state_dict": model.state_dict(),
                            "val_scores": self.val_scores,
                        }
                    )

            else:
                test_scores, test_scores_bootstrapped, test_bootstrap_samples, test_predictions = self.validate(
                    test_loader, model, criterion, status="test"
                )
                if is_best:
                    self.val_scores = val_scores
                    self.test_scores = test_scores_bootstrapped
                    self.best_epoch = self.epoch
                    self.test_bootstrap_samples = test_bootstrap_samples
                    self.predictions = test_predictions
                    self.save_checkpoint(
                            {
                                "best_epoch": self.best_epoch,
                                "state_dict": model.state_dict(),
                                "val_scores": self.val_scores,
                                "test_scores": self.test_scores,
                            }
                        )
            print(" *** best model {}".format(self.filename_best))
            scheduler.step()
            print(">>>")
            print(">>>")
            print(">>>")
            print(">>>")
            if is_best:
                self.early_stop = 0
            else:
                self.early_stop += 1
            if self.early_stop >= 10:
                print("Early stopping")
                break
        
        if self.args.num_folds > 1:
            return self.val_scores, self.best_epoch
        else:
            return self.val_scores, self.test_scores, self.best_epoch, self.test_bootstrap_samples, self.predictions

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))

        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="train epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------train epoch {}-------------------------------".format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Label) in enumerate(dataloader):
            
            data_WSI = data_WSI.float() #* Avoid error of MUSK
            
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            logit = model(data_WSI)
            loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        # calculate metrics

        scores, _, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
        return scores
    
    def bootstrap_roc_curves(self, logits, labels, n_bootstrap=1000, random_seed=42):
        """
        Bootstrap ROC curves for confidence band visualization
        Returns ROC curve coordinates for each bootstrap iteration
        """
        torch.manual_seed(random_seed + self.fold)
        
        num_classes = self.args.num_classes
        n_samples = len(labels)
        
        # Storage for bootstrap ROC curves
        bootstrap_roc_data = {
            'num_classes': num_classes,
            'bootstrap_curves': []  # Will store list of {fpr, tpr} dicts for each iteration
        }
        
        for i in range(n_bootstrap):
            bootstrap_indices = torch.randint(0, n_samples, (n_samples,), device=logits.device)
            boot_logits = logits[bootstrap_indices].cpu().numpy()
            boot_labels = labels[bootstrap_indices].cpu().numpy()
            
            iteration_curves = []
            
            if num_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(boot_labels, boot_logits[:, 1])
                iteration_curves.append({'fpr': fpr, 'tpr': tpr, 'class': 1})
            else:
                # Multiclass - one-vs-rest for each class
                for class_idx in range(num_classes):
                    y_true = (boot_labels == class_idx).astype(int)
                    y_score = boot_logits[:, class_idx]
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    iteration_curves.append({'fpr': fpr, 'tpr': tpr, 'class': class_idx})
            
            bootstrap_roc_data['bootstrap_curves'].append(iteration_curves)
        
        return bootstrap_roc_data

    def validate(self, data_loader, model, criterion, status="val"):
        model.eval()
        total_loss = 0.0
        all_logits = np.empty((0, self.args.num_classes))
        all_labels = np.empty((0, self.args.num_classes))
        all_ids = []  # ADD THIS LINE
        
        
        class_names = data_loader.dataset.classes if hasattr(data_loader.dataset, 'classes') else None


        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="{} epoch {}".format(status, self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------{} epoch {}-------------------------------".format(status, self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Label) in enumerate(dataloader):
            data_WSI = data_WSI.float()
            data_WSI = data_WSI.to(self.device)
            data_Label = F.one_hot(data_Label, num_classes=self.args.num_classes).float().to(self.device)
            with torch.no_grad():
                logit = model(data_WSI)
                loss = criterion(logit.view(1, -1), data_Label)
            # results
            all_ids.append(data_ID)  # ADD THIS LINE
            all_labels = np.row_stack((all_labels, data_Label.cpu().numpy()))
            all_logits = np.row_stack((all_logits, torch.softmax(logit, dim=-1).detach().cpu().numpy()))
            total_loss += loss.item()
        
        # calculate loss
        loss = total_loss / len(dataloader)
        print("loss: {:.4f}".format(loss))
        
        # Prepare predictions dict
        predictions_dict = {
            'case_ids': all_ids,
            'logits': all_logits,
            'labels': all_labels,
            'label_indices': np.argmax(all_labels, axis=1),
            'class_names': class_names  # ADD THIS
        }
        
        if status == "val":
            scores, _, _ = self.metrics(torch.from_numpy(all_logits).to(self.device), torch.from_numpy(all_labels).argmax(dim=1).to(self.device))
            return scores
        else:
            # For test status, compute metrics and ROC bootstraps
            general_scores, bootstrapped_scores, bootstrap_samples = self.metrics(
                torch.from_numpy(all_logits).to(self.device), 
                torch.from_numpy(all_labels).argmax(dim=1).to(self.device)
            )
            
            # Bootstrap ROC curves for confidence bands
            print("\nBootstrapping ROC curves for confidence bands...")
            roc_bootstrap_data = self.bootstrap_roc_curves(
                torch.from_numpy(all_logits).to(self.device),
                torch.from_numpy(all_labels).argmax(dim=1).to(self.device)
            )
            predictions_dict['roc_bootstrap'] = roc_bootstrap_data
            
            return general_scores, bootstrapped_scores, bootstrap_samples, predictions_dict

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        if self.test_scores is not None:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{test_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    test_score=self.test_scores["Macro_AUC_mean"],
                    epoch=self.best_epoch,
                ),
            )
        else:
            self.filename_best = os.path.join(
                self.results_dir,
                "model_best_{val_score:.4f}_{epoch}.pth.tar".format(
                    val_score=self.val_scores["Macro_AUC"],
                    epoch=self.best_epoch,
                ),
            )
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)
    
    def metrics(self, logits, labels):
        general_meter = self.meter(num_classes=self.args.num_classes, bootstrap=False)
        
        # General results
        general_results = general_meter(logits, labels)
        
        # Compute NPV from StatScores (tp, fp, tn, fn, support)
        stat_scores = general_results["StatScores"]  # Shape: [num_classes, 5]
        tp, fp, tn, fn, support = stat_scores[:, 0], stat_scores[:, 1], stat_scores[:, 2], stat_scores[:, 3], stat_scores[:, 4]
        
        # NPV = TN / (TN + FN) for each class, then macro average
        npv_per_class = tn / (tn + fn + 1e-8)  # Add small epsilon to avoid division by zero
        macro_npv = npv_per_class.mean()
        
        # Weighted NPV
        total_negatives = tn + fn
        weighted_npv = (npv_per_class * total_negatives).sum() / total_negatives.sum()
        
        # Add NPV to results
        general_results["Macro_NPV"] = macro_npv
        general_results["Weighted_NPV"] = weighted_npv
        
        del general_results["StatScores"]
        
        # Compute ECE (Expected Calibration Error)
        ece_metric = MulticlassCalibrationError(
            num_classes=self.args.num_classes, n_bins=15, norm='l1'
        ).to(self.device)
        general_results["ECE"] = ece_metric(logits, labels).item()
        
        # Compute clinical cutoff metrics
        clinical_metrics = self.compute_clinical_cutoffs(
            logits.cpu().numpy(), 
            labels.cpu().numpy()
        )
        
        # Add clinical cutoff metrics to general_results
        general_results["Sens_at_90_Spec"] = clinical_metrics["Sens_at_90_Spec"]
        general_results["Sens_at_95_Spec"] = clinical_metrics["Sens_at_95_Spec"]
        general_results["Spec_at_90_Sens"] = clinical_metrics["Spec_at_90_Sens"]
        general_results["Spec_at_95_Sens"] = clinical_metrics["Spec_at_95_Sens"]
        general_results["PPV_at_90_Spec"] = clinical_metrics["PPV_at_90_Spec"]
        general_results["NPV_at_90_Spec"] = clinical_metrics["NPV_at_90_Spec"]
        general_results["PPV_at_95_Spec"] = clinical_metrics["PPV_at_95_Spec"]
        general_results["NPV_at_95_Spec"] = clinical_metrics["NPV_at_95_Spec"]
        general_results["PPV_at_90_Sens"] = clinical_metrics["PPV_at_90_Sens"]
        general_results["NPV_at_90_Sens"] = clinical_metrics["NPV_at_90_Sens"]
        general_results["PPV_at_95_Sens"] = clinical_metrics["PPV_at_95_Sens"]
        general_results["NPV_at_95_Sens"] = clinical_metrics["NPV_at_95_Sens"]
        general_results["Per_Class_Sensitivity"] = clinical_metrics["Per_Class_Sensitivity"]
        general_results["Per_Class_Specificity"] = clinical_metrics["Per_Class_Specificity"]
        
        print("General Results:")
        print("=== Core Metrics ===")
        print(
            "Macro AUC:    {:.4f},   Macro ACC:    {:.4f},   Macro F1:    {:.4f}".format(
                general_results["Macro_AUC"],
                general_results["Macro_ACC"],
                general_results["Macro_F1"],
            )
        )
        print("=== Clinical Metrics ===")
        print(
            "Sensitivity:  {:.4f},   Specificity:  {:.4f},   PPV:         {:.4f},   NPV:         {:.4f}".format(
                general_results["Macro_Sensitivity"],
                general_results["Macro_Specificity"],
                general_results["Macro_PPV"],
                general_results["Macro_NPV"],
            )
        )
        print(
            "PR-AUC:       {:.4f},   ECE:          {:.4f}".format(
                general_results["Macro_PRAUC"],
                general_results["ECE"],
            )
        )
        print("=== Clinical Cutoff Metrics ===")
        print(
            "Sens@90%Spec: {:.4f},   Sens@95%Spec: {:.4f}".format(
                general_results["Sens_at_90_Spec"],
                general_results["Sens_at_95_Spec"],
            )
        )
        print(
            "Spec@90%Sens: {:.4f},   Spec@95%Sens: {:.4f}".format(
                general_results["Spec_at_90_Sens"],
                general_results["Spec_at_95_Sens"],
            )
        )
        print(
            "PPV@90%Spec:  {:.4f},   PPV@95%Spec:  {:.4f},   PPV@90%Sens:  {:.4f},   PPV@95%Sens:  {:.4f}".format(
                general_results["PPV_at_90_Spec"],
                general_results["PPV_at_95_Spec"],
                general_results["PPV_at_90_Sens"],
                general_results["PPV_at_95_Sens"],
            )
        )
        print(
            "NPV@90%Spec:  {:.4f},   NPV@95%Spec:  {:.4f},   NPV@90%Sens:  {:.4f},   NPV@95%Sens:  {:.4f}".format(
                general_results["NPV_at_90_Spec"],
                general_results["NPV_at_95_Spec"],
                general_results["NPV_at_90_Sens"],
                general_results["NPV_at_95_Sens"],
            )
        )
        print("=== Per-Class Sensitivity/Specificity (at optimal threshold) ===")
        for i, (sens, spec) in enumerate(zip(
            general_results["Per_Class_Sensitivity"], 
            general_results["Per_Class_Specificity"]
        )):
            print(f"  Class {i}: Sensitivity={sens:.4f}, Specificity={spec:.4f}")
        print("=== Weighted Metrics ===")
        print(
            "Weighted AUC: {:.4f},   Weighted ACC: {:.4f},   Weighted F1: {:.4f}".format(
                general_results["Weighted_AUC"],
                general_results["Weighted_ACC"],
                general_results["Weighted_F1"],
            )
        )
        
        # Custom bootstrap sampling with individual sample storage
        bootstrap_samples = self.custom_bootstrap_sampling(logits, labels)
        
        # Calculate summary statistics from bootstrap samples
        bootstrapped_results = {}
        for metric_name, samples in bootstrap_samples.items():
            #* Original
            # samples_tensor = torch.tensor(samples).clamp(0.0, 1.0)
            # bootstrapped_results[f"{metric_name}_mean"] = samples_tensor.mean().item()
            # bootstrapped_results[f"{metric_name}_ci_lower"] = torch.quantile(samples_tensor, 0.025).item()
            # bootstrapped_results[f"{metric_name}_ci_upper"] = torch.quantile(samples_tensor, 0.975).item()
            #* Updated to remove Nan
            samples_tensor = torch.tensor(samples)
            # Remove NaN values before computing statistics
            valid = samples_tensor[~torch.isnan(samples_tensor)]
            if len(valid) > 0:
                valid = valid.clamp(0.0, 1.0)
                bootstrapped_results[f"{metric_name}_mean"] = valid.mean().item()
                bootstrapped_results[f"{metric_name}_ci_lower"] = torch.quantile(valid, 0.025).item()
                bootstrapped_results[f"{metric_name}_ci_upper"] = torch.quantile(valid, 0.975).item()
            else:
                bootstrapped_results[f"{metric_name}_mean"] = float('nan')
                bootstrapped_results[f"{metric_name}_ci_lower"] = float('nan')
                bootstrapped_results[f"{metric_name}_ci_upper"] = float('nan')

        print("\nBootstrapped Results (95% CI):")
        print("=== Core Metrics ===")
        print(
            "Macro AUC:    {:.4f} ({:.4f}-{:.4f}),   Macro ACC:    {:.4f} ({:.4f}-{:.4f}),   Macro F1:    {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Macro_AUC_mean"],
                bootstrapped_results["Macro_AUC_ci_lower"],
                bootstrapped_results["Macro_AUC_ci_upper"],
                bootstrapped_results["Macro_ACC_mean"],
                bootstrapped_results["Macro_ACC_ci_lower"],
                bootstrapped_results["Macro_ACC_ci_upper"],
                bootstrapped_results["Macro_F1_mean"],
                bootstrapped_results["Macro_F1_ci_lower"],
                bootstrapped_results["Macro_F1_ci_upper"],
            )
        )
        print("=== Clinical Metrics ===")
        print(
            "Sensitivity:  {:.4f} ({:.4f}-{:.4f}),   Specificity:  {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Macro_Sensitivity_mean"],
                bootstrapped_results["Macro_Sensitivity_ci_lower"],
                bootstrapped_results["Macro_Sensitivity_ci_upper"],
                bootstrapped_results["Macro_Specificity_mean"],
                bootstrapped_results["Macro_Specificity_ci_lower"],
                bootstrapped_results["Macro_Specificity_ci_upper"],
            )
        )
        print(
            "PPV:          {:.4f} ({:.4f}-{:.4f}),   NPV:          {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Macro_PPV_mean"],
                bootstrapped_results["Macro_PPV_ci_lower"],
                bootstrapped_results["Macro_PPV_ci_upper"],
                bootstrapped_results["Macro_NPV_mean"],
                bootstrapped_results["Macro_NPV_ci_lower"],
                bootstrapped_results["Macro_NPV_ci_upper"],
            )
        )
        print(
            "PR-AUC:       {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Macro_PRAUC_mean"],
                bootstrapped_results["Macro_PRAUC_ci_lower"],
                bootstrapped_results["Macro_PRAUC_ci_upper"],
            )
        )
        print("=== Clinical Cutoff Metrics ===")
        print(
            "Sens@90%Spec: {:.4f} ({:.4f}-{:.4f}),   Sens@95%Spec: {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Sens_at_90_Spec_mean"],
                bootstrapped_results["Sens_at_90_Spec_ci_lower"],
                bootstrapped_results["Sens_at_90_Spec_ci_upper"],
                bootstrapped_results["Sens_at_95_Spec_mean"],
                bootstrapped_results["Sens_at_95_Spec_ci_lower"],
                bootstrapped_results["Sens_at_95_Spec_ci_upper"],
            )
        )
        print(
            "Spec@90%Sens: {:.4f} ({:.4f}-{:.4f}),   Spec@95%Sens: {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["Spec_at_90_Sens_mean"],
                bootstrapped_results["Spec_at_90_Sens_ci_lower"],
                bootstrapped_results["Spec_at_90_Sens_ci_upper"],
                bootstrapped_results["Spec_at_95_Sens_mean"],
                bootstrapped_results["Spec_at_95_Sens_ci_lower"],
                bootstrapped_results["Spec_at_95_Sens_ci_upper"],
            )
        )
        print(
            "PPV@90%Spec:  {:.4f} ({:.4f}-{:.4f}),   PPV@95%Spec:  {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["PPV_at_90_Spec_mean"],
                bootstrapped_results["PPV_at_90_Spec_ci_lower"],
                bootstrapped_results["PPV_at_90_Spec_ci_upper"],
                bootstrapped_results["PPV_at_95_Spec_mean"],
                bootstrapped_results["PPV_at_95_Spec_ci_lower"],
                bootstrapped_results["PPV_at_95_Spec_ci_upper"],
            )
        )
        print(
            "PPV@90%Sens:  {:.4f} ({:.4f}-{:.4f}),   PPV@95%Sens:  {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["PPV_at_90_Sens_mean"],
                bootstrapped_results["PPV_at_90_Sens_ci_lower"],
                bootstrapped_results["PPV_at_90_Sens_ci_upper"],
                bootstrapped_results["PPV_at_95_Sens_mean"],
                bootstrapped_results["PPV_at_95_Sens_ci_lower"],
                bootstrapped_results["PPV_at_95_Sens_ci_upper"],
            )
        )
        print(
            "NPV@90%Spec:  {:.4f} ({:.4f}-{:.4f}),   NPV@95%Spec:  {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["NPV_at_90_Spec_mean"],
                bootstrapped_results["NPV_at_90_Spec_ci_lower"],
                bootstrapped_results["NPV_at_90_Spec_ci_upper"],
                bootstrapped_results["NPV_at_95_Spec_mean"],
                bootstrapped_results["NPV_at_95_Spec_ci_lower"],
                bootstrapped_results["NPV_at_95_Spec_ci_upper"],
            )
        )
        print(
            "NPV@90%Sens:  {:.4f} ({:.4f}-{:.4f}),   NPV@95%Sens:  {:.4f} ({:.4f}-{:.4f})".format(
                bootstrapped_results["NPV_at_90_Sens_mean"],
                bootstrapped_results["NPV_at_90_Sens_ci_lower"],
                bootstrapped_results["NPV_at_90_Sens_ci_upper"],
                bootstrapped_results["NPV_at_95_Sens_mean"],
                bootstrapped_results["NPV_at_95_Sens_ci_lower"],
                bootstrapped_results["NPV_at_95_Sens_ci_upper"],
            )
        )
        
        # Add ECE (point estimate, not bootstrapped) to bootstrapped_results
        bootstrapped_results["ECE"] = general_results["ECE"]
        
        return general_results, bootstrapped_results, bootstrap_samples

    # Update custom_bootstrap_sampling to include new metrics
    def custom_bootstrap_sampling(self, logits, labels, n_bootstrap=1000, random_seed=42):
        """
        Custom bootstrap sampling that returns individual samples for statistical testing
        """
        torch.manual_seed(random_seed + self.fold)
        
        # Initialize storage for bootstrap samples (including new metrics)
        bootstrap_samples = {
            "Macro_AUC": [], "Macro_ACC": [], "Macro_F1": [],
            "Weighted_AUC": [], "Weighted_ACC": [], "Weighted_F1": [],
            "Macro_Sensitivity": [], "Macro_Specificity": [], "Macro_PPV": [], "Macro_NPV": [], "Macro_PRAUC": [],
            "Weighted_Sensitivity": [], "Weighted_Specificity": [], "Weighted_PPV": [], "Weighted_NPV": [], "Weighted_PRAUC": [],
            # Clinical cutoff metrics
            "Sens_at_90_Spec": [], "Sens_at_95_Spec": [],
            "Spec_at_90_Sens": [], "Spec_at_95_Sens": [],
            # PPV/NPV at clinical cutoffs
            "PPV_at_90_Spec": [], "NPV_at_90_Spec": [],
            "PPV_at_95_Spec": [], "NPV_at_95_Spec": [],
            "PPV_at_90_Sens": [], "NPV_at_90_Sens": [],
            "PPV_at_95_Sens": [], "NPV_at_95_Sens": []
        }
        
        meter = self.meter(num_classes=self.args.num_classes, bootstrap=False)
        n_samples = len(labels)
        
        for i in range(n_bootstrap):
            bootstrap_indices = torch.randint(0, n_samples, (n_samples,), device=logits.device)
            boot_logits = logits[bootstrap_indices]
            boot_labels = labels[bootstrap_indices]
            
            boot_results = meter(boot_logits, boot_labels)
            
            # Compute NPV for this bootstrap sample
            stat_scores = boot_results["StatScores"]
            tp, fp, tn, fn, support = stat_scores[:, 0], stat_scores[:, 1], stat_scores[:, 2], stat_scores[:, 3], stat_scores[:, 4]
            npv_per_class = tn / (tn + fn + 1e-8)
            macro_npv = npv_per_class.mean()
            total_negatives = tn + fn
            weighted_npv = (npv_per_class * total_negatives).sum() / total_negatives.sum()
            
            # Compute clinical cutoff metrics for this bootstrap sample
            clinical_metrics = self.compute_clinical_cutoffs(
                boot_logits.cpu().numpy(),
                boot_labels.cpu().numpy()
            )
            
            # Store results
            for metric_name in bootstrap_samples.keys():
                if metric_name in clinical_metrics:
                    # Clinical cutoff metrics (including NPV_at_X, PPV_at_X)
                    bootstrap_samples[metric_name].append(clinical_metrics[metric_name])
                elif "NPV" in metric_name:
                    if "Macro" in metric_name:
                        bootstrap_samples[metric_name].append(macro_npv.item())
                    else:
                        bootstrap_samples[metric_name].append(weighted_npv.item())
                else:
                    bootstrap_samples[metric_name].append(boot_results[metric_name].item())
        
        return bootstrap_samples

    def meter(self, num_classes, bootstrap=False):
        metrics: Dict[str, Metric] = {
            # Existing metrics
            "Macro_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_F1": F1Score(num_classes=int(num_classes), average="macro", task="multiclass").to(self.device),
            "Macro_AUC": AUROC(num_classes=num_classes, average="macro", task="multiclass").to(self.device),
            "Weighted_ACC": MulticlassAccuracy(top_k=1, num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_F1": F1Score(num_classes=int(num_classes), average="weighted", task="multiclass").to(self.device),
            "Weighted_AUC": AUROC(num_classes=num_classes, average="weighted", task="multiclass").to(self.device),
            
            # New clinical metrics
            "Macro_Sensitivity": MulticlassRecall(num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_Specificity": MulticlassSpecificity(num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_PPV": MulticlassPrecision(num_classes=int(num_classes), average="macro").to(self.device),
            "Macro_PRAUC": AveragePrecision(num_classes=int(num_classes), average="macro", task="multiclass").to(self.device),
            
            "Weighted_Sensitivity": MulticlassRecall(num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_Specificity": MulticlassSpecificity(num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_PPV": MulticlassPrecision(num_classes=int(num_classes), average="weighted").to(self.device),
            "Weighted_PRAUC": AveragePrecision(num_classes=int(num_classes), average="weighted", task="multiclass").to(self.device),
            
            # StatScores for computing NPV manually
            "StatScores": MulticlassStatScores(num_classes=int(num_classes), average=None).to(self.device),
        }
        
        # Bootstrap wrap
        if bootstrap:
            for k, m in metrics.items():
                metrics[k] = BootStrapper(m, num_bootstraps=3000, sampling_strategy="multinomial").to(self.device)
        
        metrics = MetricCollection(metrics)
        return metrics
