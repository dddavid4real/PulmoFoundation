"""
Gigapath Engine for Survival Prediction

Adapted from AttMIL engine to handle coordinate inputs.
Handles (ID, Slide, Coords, Event, Censorship, Label) tuples.
"""

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
    def __init__(self, args, results_dir, fold=0):
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
        self.best_bootstrap_samples = None

    def find_latest_checkpoint(self, root_dir, model_name):
        """Find the latest checkpoint directory for a given model name."""
        if not os.path.isdir(root_dir):
            print(f"[ERROR] Root directory does not exist: {root_dir}")
            return None
        
        matching_dirs = []
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path) and item.startswith(f"[{model_name}]-"):
                matching_dirs.append(item)
        
        if not matching_dirs:
            print(f"[WARNING] No matching directories found for model [{model_name}]")
            return None
        
        def parse_dir_datetime(dir_name):
            try:
                pattern = r'\[([^\]]+)\]-\[([^\]]+)\]-\[([^\]]+)\]'
                match = re.match(pattern, dir_name)
                if match:
                    _, date_part, time_part = match.groups()
                    time_formatted = time_part.replace('-', ':')
                    datetime_str = f"{date_part} {time_formatted}"
                    return datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print(f"[ERROR] Failed to parse datetime for {dir_name}: {e}")
            return datetime(1900, 1, 1)
        
        matching_dirs.sort(key=parse_dir_datetime, reverse=True)
        latest_dir = matching_dirs[0]
        latest_dir_path = os.path.join(root_dir, latest_dir)
        
        # Find checkpoint files
        search_paths = [latest_dir_path, os.path.join(latest_dir_path, 'fold_0')]
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
            try:
                all_files = os.listdir(search_path)
            except:
                continue
            
            model_best_files = [os.path.join(search_path, f) for f in all_files 
                               if f.endswith('.pth.tar') and f.startswith('model_best_')]
            if model_best_files:
                return model_best_files[0]
            
            pth_tar_files = [os.path.join(search_path, f) for f in all_files 
                            if f.endswith('.pth.tar')]
            if pth_tar_files:
                return pth_tar_files[0]
        
        return None

    def learning(self, model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        
        self.epoch = 0
        
        # Handle checkpoint loading
        if self.args.resume is not None:
            if os.path.isdir(self.args.resume):
                checkpoint_key = f"{self.args.model}-{self.args.feature}"
                checkpoint_path = self.find_latest_checkpoint(self.args.resume, checkpoint_key)
                if checkpoint_path and os.path.isfile(checkpoint_path):
                    print("=> loading checkpoint '{}'".format(checkpoint_path))
                    checkpoint = torch.load(checkpoint_path)
                    self.best_scores = checkpoint['best_score']
                    model.load_state_dict(checkpoint['state_dict'])
                    print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            elif os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))

        # Evaluation mode
        if self.args.evaluate:
            print("=> Running evaluation mode on external dataset")
            c_index, patient_results, bootstrap_samples = self.validate_with_bootstrap(val_loader, model, criterion)
            
            c_index_samples = np.array(bootstrap_samples["C_Index"])
            ci_lower = np.percentile(c_index_samples, 2.5)
            ci_upper = np.percentile(c_index_samples, 97.5)
            mean_c_index = c_index_samples.mean()
            
            print(' *** Evaluation C-index: {:.4f} ({:.4f}, {:.4f})'.format(mean_c_index, ci_lower, ci_upper))
            
            study_prefix = self.args.study if hasattr(self.args, 'study') and self.args.study else None
            self.save_pkl(patient_results, prefix=study_prefix)
            self.save_predictions_csv(patient_results, prefix=study_prefix)
            
            return mean_c_index, ci_lower, ci_upper, bootstrap_samples

        # Training loop
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            self.train(train_loader, model, criterion, optimizer)
            scores, results = self.validate(val_loader, model, criterion)
            
            is_best = scores > self.best_scores
            if is_best:
                self.best_scores = scores
                self.best_epoch = self.epoch
                _, _, bootstrap_samples = self.validate_with_bootstrap(val_loader, model, criterion)
                self.best_bootstrap_samples = bootstrap_samples
                
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_scores
                })
                self.save_pkl(results)

            print(' *** best eval score={:.4f} at epoch {}'.format(self.best_scores, self.best_epoch))
            scheduler.step()
            print('>>>')
            print('>>>')
        
        # Testing
        print('start testing')
        model.load_state_dict(torch.load(self.filename_best)['state_dict'])
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        _, test_patient_results = self.validate(test_loader, model, criterion)
        self.save_pkl(test_patient_results, prefix='Internal')
        self.save_predictions_csv(test_patient_results, prefix='Internal')

        # Bootstrap for final test results (coordinate-aware, no bootstrap_survivalv2)
        model.eval()
        all_risk_scores = np.zeros((len(test_loader)))
        all_censorships = np.zeros((len(test_loader)))
        all_event_times = np.zeros((len(test_loader)))
        for batch_idx, (data_ID, data_WSI, data_Coords, data_Event, data_Censorship, data_Label) in enumerate(test_loader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda().float()
                data_Coords = data_Coords.cuda().float()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                hazards, S = model(data_WSI, data_Coords)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        
        bootstrap_samples = self.custom_bootstrap_sampling(all_risk_scores, all_censorships, all_event_times, n_bootstrap=1000)
        c_index_samples = np.array(bootstrap_samples["C_Index"])
        mean_c_index = c_index_samples.mean()
        ci_lower = np.percentile(c_index_samples, 2.5)
        ci_upper = np.percentile(c_index_samples, 97.5)
        
        print(' *** test c-index: {:.4f} ({:.4f}, {:.4f})'.format(mean_c_index, ci_lower, ci_upper))
        return mean_c_index, ci_lower, ci_upper

    def learning_with_bootstrap(self, model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler):
        """Same as learning but returns bootstrap samples as 4th value"""
        result = self.learning(model, train_loader, val_loader, test_dataset, criterion, optimizer, scheduler)
        
        if self.args.evaluate and result is not None and len(result) == 4:
            return result
        elif result is not None:
            mean_c_index, ci_lower, ci_upper = result
            return mean_c_index, ci_lower, ci_upper, self.best_bootstrap_samples
        else:
            return None, None, None, None

    def train(self, data_loader, model, criterion, optimizer):
        """Train one epoch - handles (ID, Slide, Coords, Event, Censorship, Label) tuples."""
        model.train()

        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch {}'.format(self.epoch))
        
        for batch_idx, (data_ID, data_WSI, data_Coords, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda().float()
                data_Coords = data_Coords.cuda().float()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            
            # Forward with coordinates
            hazards, S = model(data_WSI, data_Coords)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('train/loss', loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

    def validate(self, data_loader, model, criterion):
        """Validate - handles (ID, Slide, Coords, Event, Censorship, Label) tuples."""
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        patient_results = {}
        dataloader = tqdm(data_loader, desc='Test Epoch {}'.format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Coords, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda().float()
                data_Coords = data_Coords.cuda().float()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            
            with torch.no_grad():
                hazards, S = model(data_WSI, data_Coords)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            total_loss += loss.item()
            
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            
            slide_id = data_ID[0]
            patient_results.update({
                slide_id: {
                    'slide_id': np.array(slide_id), 
                    'risk': risk, 
                    'disc_label': data_Label.item(), 
                    'survival': data_Event, 
                    'censorship': data_Censorship
                }
            })
        
        loss = total_loss / len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', loss, self.epoch)
            self.writer.add_scalar('val/c_index', c_index, self.epoch)
            
        return c_index, patient_results

    def validate_with_bootstrap(self, data_loader, model, criterion):
        """Same as validate but also returns bootstrap samples"""
        c_index, patient_results = self.validate(data_loader, model, criterion)
        
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        
        model.eval()
        for batch_idx, (data_ID, data_WSI, data_Coords, data_Event, data_Censorship, data_Label) in enumerate(data_loader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda().float()
                data_Coords = data_Coords.cuda().float()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            with torch.no_grad():
                hazards, S = model(data_WSI, data_Coords)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        
        bootstrap_samples = self.custom_bootstrap_sampling(all_risk_scores, all_censorships, all_event_times)
        return c_index, patient_results, bootstrap_samples

    def custom_bootstrap_sampling(self, risk_scores, censorships, event_times, n_bootstrap=1000, random_seed=42):
        """Custom bootstrap sampling for C-index"""
        np.random.seed(random_seed + self.fold)
        bootstrap_samples = {"C_Index": []}
        n_samples = len(risk_scores)
        
        for i in range(n_bootstrap):
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_risk_scores = risk_scores[bootstrap_indices]
            boot_censorships = censorships[bootstrap_indices]
            boot_event_times = event_times[bootstrap_indices]
            
            try:
                boot_c_index = concordance_index_censored(
                    (1 - boot_censorships).astype(bool), 
                    boot_event_times, 
                    boot_risk_scores, 
                    tied_tol=1e-08
                )[0]
                bootstrap_samples["C_Index"].append(boot_c_index)
            except:
                boot_c_index = concordance_index_censored(
                    (1 - censorships).astype(bool), 
                    event_times, 
                    risk_scores, 
                    tied_tol=1e-08
                )[0]
                bootstrap_samples["C_Index"].append(boot_c_index)
        
        samples_array = np.array(bootstrap_samples["C_Index"])
        print("Bootstrap C-Index: {:.4f}±{:.4f}".format(samples_array.mean(), samples_array.std()))
        return bootstrap_samples

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(
            self.results_dir,
            'fold_' + str(self.fold),
            'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch'])
        )
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def save_pkl(self, results, prefix=None):
        if prefix:
            results_pkl_path = os.path.join(self.results_dir, f'{prefix}_predictions.pkl')
        else:
            results_pkl_path = os.path.join(self.results_dir, f'split_{self.fold}_results.pkl')
        
        with open(results_pkl_path, 'wb') as writer:
            pickle.dump(results, writer)
        print(f"[Saved] Patient results (pickle): {results_pkl_path}")

    def save_predictions_csv(self, patient_results, prefix=None):
        """Save patient-level predictions to CSV for KM curve plotting"""
        import csv
        
        if prefix:
            csv_path = os.path.join(self.results_dir, f'{prefix}_predictions.csv')
        else:
            csv_path = os.path.join(self.results_dir, f'predictions_split_{self.fold}.csv')
        
        all_risk_scores = []
        for patient_id, data in patient_results.items():
            risk = data['risk'][0] if isinstance(data['risk'], np.ndarray) else data['risk']
            all_risk_scores.append(risk)
        
        median_risk = np.median(all_risk_scores)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Patient_ID', 'Risk_Score', 'Survival_Time', 'Event_Occurred', 'Censorship', 'Discrete_Label', 'Stratification'])
            
            for patient_id, data in patient_results.items():
                risk = data['risk'][0] if isinstance(data['risk'], np.ndarray) else data['risk']
                survival = data['survival'] if isinstance(data['survival'], (int, float)) else data['survival'].item()
                censorship = data['censorship'].item() if hasattr(data['censorship'], 'item') else data['censorship']
                event_occurred = 1 - censorship
                disc_label = data['disc_label']
                stratification = "High Risk" if risk >= median_risk else "Low Risk"
                
                writer.writerow([patient_id, f"{risk:.6f}", survival, event_occurred, censorship, disc_label, stratification])
        
        print(f"[Saved] Predictions CSV: {csv_path}")
