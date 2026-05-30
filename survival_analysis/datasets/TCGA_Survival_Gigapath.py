"""
TCGA Survival Dataset for Gigapath

Extended from TCGA_Survival to load coordinate files for Gigapath model.
Returns (ID, WSI, Coords, Event, Censorship, Label) tuples.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import h5py


class TCGA_Survival_Gigapath(data.Dataset):
    def __init__(self, csv_file, feature_path, modal, study, feature):
        self.modal = modal
        self.final_feature_path = os.path.join(feature_path, f'{feature}')
        self.coord_path = feature_path.replace("/pt_files", "/patches").replace("\\pt_files", "\\patches")
        
        print(f'feature path: {feature_path}')
        print(f'feature: {feature}')
        print(f'modal: {modal}')
        print(f'study: {study}')
        print(f'coord path: {self.coord_path}')
        
        print('[dataset] loading dataset from %s' % (csv_file))
        rows = pd.read_csv(csv_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(self.rows)))

    def get_split(self):
        train_split = self.rows[self.rows['split'] == 'train'].index.tolist()
        val_split = self.rows[self.rows['split'] == 'validation'].index.tolist()
        test_split = self.rows[self.rows['split'] == 'test'].index.tolist()
        return train_split, val_split, test_split

    def read_WSI(self, WSI):
        """Read WSI features from .pt files"""
        wsi = [torch.load(os.path.join(self.final_feature_path, x)) for x in WSI.split(';')]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def read_coords(self, WSI, n_patches):
        """Read coordinates from h5 files"""
        wsi_names = WSI.split(';')
        all_coords = []
        
        for wsi_name in wsi_names:
            # Try with .pt extension removed
            base_name = wsi_name.replace('.pt', '')
            h5_path = os.path.join(self.coord_path, f'{base_name}.h5')
            
            if os.path.exists(h5_path):
                try:
                    with h5py.File(h5_path, 'r') as f:
                        coords = f['coords'][:]
                        all_coords.append(torch.from_numpy(coords).float())
                except Exception as e:
                    print(f"[Warning] Error reading coords from {h5_path}: {e}")
                    # Generate dummy coords for this WSI
                    all_coords.append(self._generate_dummy_coords(n_patches // len(wsi_names)))
            else:
                print(f"[Warning] Coord file not found: {h5_path}")
                all_coords.append(self._generate_dummy_coords(n_patches // len(wsi_names)))
        
        if all_coords:
            coords = torch.cat(all_coords, dim=0)
        else:
            coords = self._generate_dummy_coords(n_patches)
        
        # Ensure coords match feature count
        if coords.shape[0] != n_patches:
            print(f"[Warning] Coord shape mismatch: {coords.shape[0]} vs {n_patches} patches")
            coords = self._generate_dummy_coords(n_patches)
        
        return coords

    def _generate_dummy_coords(self, n_patches):
        """Generate normalized dummy coordinates as a grid pattern"""
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        coords = []
        for i in range(n_patches):
            x = (i % grid_size) / grid_size
            y = (i // grid_size) / grid_size
            coords.append([x, y])
        return torch.tensor(coords, dtype=torch.float32)

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI = case[:5]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        
        if self.modal == 'WSI':
            WSI_features = self.read_WSI(WSI)
            n_patches = WSI_features.shape[0]
            Coords = self.read_coords(WSI, n_patches)
            return (ID, WSI_features, Coords, Event, Censorship, Label)
        else:
            raise NotImplementedError('modality [{}] is not implemented'.format(self.modal))

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows
