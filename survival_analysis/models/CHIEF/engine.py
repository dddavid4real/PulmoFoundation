import os
import numpy as np
from tqdm import tqdm
import pickle
import re
from datetime import datetime
from sksurv.metrics import concordance_index_censored
from utils.bootstrap import bootstrap_survivalv2
import torch.optim
import torch


class Engine(object):
    def __init__(self, args, results_dir, fold = 0):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_scores = 0
        self.best_epoch = 0
        self.filename_best = None
        self.best_bootstrap_samples = None  # NEW: Store bootstrap samples from best model

    def find_latest_checkpoint(self, root_dir, model_name):
        """
        Find the latest checkpoint directory for a given model name.
        Directory format: [model_name-feature]-[YYYY-MM-DD]-[HH-MM-SS]
        
        Args:
            root_dir: Root directory containing checkpoint folders
            model_name: Model name with feature (e.g., 'AttMIL-chief')
        
        Returns:
            Path to the checkpoint file, or None if not found
        """
        if not os.path.isdir(root_dir):
            print(f"[ERROR] Root directory does not exist: {root_dir}")
            return None
        
        # Find all directories that match the pattern [model_name]-[date]-[time]
        matching_dirs = []
        
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item.startswith(f"[{model_name}]-"):
                matching_dirs.append(item)
                print(f"[DEBUG] Matched folder: {item}")
        
        if not matching_dirs:
            print(f"[WARNING] No matching directories found for model [{model_name}]")
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
                    parsed = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                    print(f"[DEBUG] Parsed datetime for {dir_name}: {parsed}")
                    return parsed
                else:
                    print(f"[DEBUG] Regex did not match folder name: {dir_name}")
                    return datetime(1900, 1, 1)
            except Exception as e:
                print(f"[ERROR] Failed to parse datetime for {dir_name}: {e}")
                return datetime(1900, 1, 1)
        
        # Sort by datetime (most recent first)
        matching_dirs.sort(key=parse_dir_datetime, reverse=True)
        latest_dir = matching_dirs[0]
        latest_dir_path = os.path.join(root_dir, latest_dir)
        
        print(f"[INFO] Latest checkpoint directory selected: {latest_dir_path}")
        
        # Find checkpoint files in the directory (including fold_0 subfolder if exists)
        search_paths = [latest_dir_path, os.path.join(latest_dir_path, 'fold_0')]
        
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            try:
                all_files = os.listdir(search_path)
            except:
                continue
            
            # Filter files by extension
            pth_tar_files = []
            model_best_files = []
            
            for file in all_files:
                file_path = os.path.join(search_path, file)
                if file.endswith('.pth.tar'):
                    pth_tar_files.append(file_path)
                    if file.startswith('model_best_'):
                        model_best_files.append(file_path)
            
            # Prefer model_best files, then any pth.tar files
            if model_best_files:
                print(f"[INFO] Found checkpoint: {model_best_files[0]}")
                return model_best_files[0]
            elif pth_tar_files:
                print(f"[INFO] Found checkpoint: {pth_tar_files[0]}")
                return pth_tar_files[0]
        
        print(f"[ERROR] No checkpoint file found in {latest_dir_path}")
        return None

    # ORIGINAL METHOD - UNCHANGED
    def learning(self, model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        
        self.epoch = 0
        
        if self.args.resume is not None:
            # Check if it's a directory (evaluation mode) or file (training resume)
            if os.path.isdir(self.args.resume):
                # Find checkpoint in directory based on model and feature name
                checkpoint_key = f"{self.args.model}-{self.args.feature}"
                checkpoint_path = self.find_latest_checkpoint(self.args.resume, checkpoint_key)
                
                if checkpoint_path and os.path.isfile(checkpoint_path):
                    print("=> loading checkpoint '{}'".format(checkpoint_path))
                    checkpoint = torch.load(checkpoint_path)
                    self.best_scores = checkpoint['best_score']
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
                else:
                    print("=> no checkpoint found for model '{}' in directory '{}'".format(checkpoint_key, self.args.resume))
            elif os.path.isfile(self.args.resume):
                # Original file-based loading
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            print("=> Running evaluation mode on external dataset")
            # Run validation with bootstrap on the external test set
            c_index, patient_results, bootstrap_samples = self.validate_with_bootstrap(val_loader, model, criterion)
            
            # Calculate confidence intervals from bootstrap samples
            c_index_samples = np.array(bootstrap_samples["C_Index"])
            ci_lower = np.percentile(c_index_samples, 2.5)
            ci_upper = np.percentile(c_index_samples, 97.5)
            mean_c_index = c_index_samples.mean()
            
            print(' *** Evaluation C-index: {:.4f} ({:.4f}, {:.4f})'.format(mean_c_index, ci_lower, ci_upper))
            
            # patient_results contains all data needed: ID, risk, survival time, censorship
            study_prefix = self.args.study if hasattr(self.args, 'study') and self.args.study else None
            self.save_pkl(patient_results, prefix=study_prefix)
            self.save_predictions_csv(patient_results, prefix=study_prefix)
            
            return mean_c_index, ci_lower, ci_upper, bootstrap_samples
        

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            scores,results = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = scores > self.best_scores
            if is_best:
                self.best_scores = scores
                self.best_epoch = self.epoch
                # NEW: Also collect bootstrap samples for best model
                _, _, bootstrap_samples = self.validate_with_bootstrap(val_loader, model, criterion)
                self.best_bootstrap_samples = bootstrap_samples
                
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_scores})
                
                self.save_pkl(results)

            print(' *** best eval score={:.4f} at epoch {}'.format(self.best_scores, self.best_epoch))
            scheduler.step()
            print('>>>')
            print('>>>')
        print('start testing')
        model.load_state_dict(torch.load(self.filename_best)['state_dict'])
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        _, test_patient_results = self.validate(test_loader, model, criterion)

        # Save test set predictions
        self.save_pkl(test_patient_results, prefix='Internal')
        self.save_predictions_csv(test_patient_results, prefix='Internal')

        import time
        max_attempts = 100  # safety limit to avoid infinite loop
        attempt = 0
        success = False
        while attempt < max_attempts:
            try:
                mean_c_index, ci_lower, ci_upper = bootstrap_survivalv2(
                    model, test_dataset, n_iterations=1000, device='cuda'
                )
                success = True
                break  # If successful, exit loop
            except Exception as e:
                print(f"[Failed on attempt {attempt + 1}]")
                attempt += 1
                time.sleep(0.1)  # short delay before retry
        if not success:
            raise RuntimeError("Bootstrap failed too many times due to class imbalance in sampling.")
        print(' *** test c-index: {:.4f} ({:.4f}, {:.4f})'.format(mean_c_index, ci_lower, ci_upper))

        return mean_c_index, ci_lower, ci_upper

    def learning_with_bootstrap(self, model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler):
        """Same as learning but returns bootstrap samples as 4th value"""
        result = self.learning(model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler)
        
        # In evaluation mode, learning() returns 4 values already
        if self.args.evaluate and result is not None and len(result) == 4:
            return result
        # In training mode, learning() returns 3 values, we add bootstrap samples
        elif result is not None:
            mean_c_index, ci_lower, ci_upper = result
            return mean_c_index, ci_lower, ci_upper, self.best_bootstrap_samples
        else:
            # Shouldn't happen but handle gracefully
            return None, None, None, None

    def train(self, data_loader, model, criterion, optimizer):
        model.train()

        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch {}'.format(self.epoch))
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_WSI = data_WSI.float() #* Avoid error of MUSK
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            hazards, S = model(data_WSI)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        patient_results = {}
        dataloader = tqdm(data_loader, desc='Test Epoch {}'.format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_WSI = data_WSI.float() #* Avoid error of MUSK
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            with torch.no_grad():
                hazards, S = model(data_WSI)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            total_loss += loss.item()
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            slide_id = data_ID[0]
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': data_Label.item(), 'survival': data_Event, 'censorship': data_Censorship}})
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', loss, self.epoch)
            self.writer.add_scalar('val/c_index', c_index, self.epoch)
            
        return c_index, patient_results

    def validate_with_bootstrap(self, data_loader, model, criterion):
        """Same as validate but also returns bootstrap samples as 3rd value"""
        c_index, patient_results = self.validate(data_loader, model, criterion)
        
        # Get the data again for bootstrap sampling
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        
        model.eval()
        for batch_idx, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(data_loader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_WSI = data_WSI.float()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                hazards, S = model(data_WSI)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        
        bootstrap_samples = self.custom_bootstrap_sampling(all_risk_scores, all_censorships, all_event_times)
        return c_index, patient_results, bootstrap_samples

    def custom_bootstrap_sampling(self, risk_scores, censorships, event_times, n_bootstrap=1000, random_seed=42):
        """
        Custom bootstrap sampling for C-index that returns individual samples for statistical testing
        """
        # Set random seed for reproducible bootstrap sampling across models
        np.random.seed(random_seed + self.fold)
        
        bootstrap_samples = {
            "C_Index": []
        }
        
        n_samples = len(risk_scores)
        
        for i in range(n_bootstrap):
            # Generate bootstrap indices
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Sample with replacement
            boot_risk_scores = risk_scores[bootstrap_indices]
            boot_censorships = censorships[bootstrap_indices]
            boot_event_times = event_times[bootstrap_indices]
            
            # Calculate C-index for this bootstrap sample
            try:
                boot_c_index = concordance_index_censored(
                    (1 - boot_censorships).astype(bool), 
                    boot_event_times, 
                    boot_risk_scores, 
                    tied_tol=1e-08
                )[0]
                bootstrap_samples["C_Index"].append(boot_c_index)
            except:
                # If calculation fails, use original C-index
                boot_c_index = concordance_index_censored(
                    (1 - censorships).astype(bool), 
                    event_times, 
                    risk_scores, 
                    tied_tol=1e-08
                )[0]
                bootstrap_samples["C_Index"].append(boot_c_index)
        
        # Print bootstrap results
        samples_array = np.array(bootstrap_samples["C_Index"])
        print("Bootstrap C-Index: {:.4f}±{:.4f}".format(samples_array.mean(), samples_array.std()))
        
        return bootstrap_samples

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def save_pkl(self, results, prefix=None):
        """
        Save results to pickle file
        
        Args:
            results: Dictionary of patient results
            prefix: Optional prefix for filename (e.g., study name for external validation)
        """
        if prefix:
            # External validation: {study}_predictions.pkl
            results_pkl_path = os.path.join(self.results_dir, f'{prefix}_predictions.pkl')
        else:
            # Training: split_0_results.pkl (original naming)
            results_pkl_path = os.path.join(self.results_dir, f'split_{self.fold}_results.pkl')
        
        with open(results_pkl_path, 'wb') as writer:
            pickle.dump(results, writer)
        print(f"[Saved] Patient results (pickle): {results_pkl_path}")
        
    def save_predictions_csv(self, patient_results, prefix=None):
        """
        Save patient-level predictions to CSV for KM curve plotting
        
        Args:
            patient_results: Dictionary of patient results
            prefix: Optional prefix for filename (e.g., study name for external validation)
        
        Columns: ID, Risk_Score, Survival_Time, Event_Occurred, Censorship, Discrete_Label, Stratification
        """
        import csv
        
        if prefix:
            # External validation: {study}_predictions.csv
            csv_path = os.path.join(self.results_dir, f'{prefix}_predictions.csv')
        else:
            # Training: predictions_split_0.csv (original naming)
            csv_path = os.path.join(self.results_dir, f'predictions_split_{self.fold}.csv')
        
        # Calculate median risk score for stratification
        all_risk_scores = []
        for patient_id, data in patient_results.items():
            risk = data['risk'][0] if isinstance(data['risk'], np.ndarray) else data['risk']
            all_risk_scores.append(risk)
        
        median_risk = np.median(all_risk_scores)
        print(f"[INFO] Risk stratification threshold (median): {median_risk:.6f}")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header - Added "Stratification" column
            writer.writerow(['Patient_ID', 'Risk_Score', 'Survival_Time', 'Event_Occurred', 'Censorship', 'Discrete_Label', 'Stratification'])
            
            # Data rows
            for patient_id, data in patient_results.items():
                risk = data['risk'][0] if isinstance(data['risk'], np.ndarray) else data['risk']
                survival = data['survival'] if isinstance(data['survival'], (int, float)) else data['survival'].item()
                censorship = data['censorship'].item() if hasattr(data['censorship'], 'item') else data['censorship']
                event_occurred = 1 - censorship  # Convert censorship to event indicator
                disc_label = data['disc_label']
                
                # Stratify based on median risk score
                stratification = "High Risk" if risk >= median_risk else "Low Risk"
                
                writer.writerow([
                    patient_id,
                    f"{risk:.6f}",
                    survival,
                    event_occurred,
                    censorship,
                    disc_label,
                    stratification
                ])
        
        print(f"[Saved] Predictions CSV for KM curves: {csv_path}")