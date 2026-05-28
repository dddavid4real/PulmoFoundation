import os
import h5py
import pandas as pd

from tqdm import tqdm

import torch
import torch.utils.data as data


class Dataset_Subtyping_Gigapath(data.Dataset):
    """
    Dataset class for GigaPath model that loads both features (.pt) and coordinates (.h5).
    
    The h5_files path is derived from pt_files path by replacing 'pt_files' with 'patches'.
    Note: h5 files are directly in the patches folder, no feature subfolder.
    
    Path structure:
        pt_files: {root}/{feature}/{slide}.pt
        h5_files: {patches_root}/{slide}.h5  (no feature subfolder)
    """
    
    def __init__(self, root, csv_file, feature):
        self.feature = feature
        # Handle multiple roots (comma-separated)
        if "," in root:
            self.root = root.split(",")
        else:
            self.root = [root]
        
        # Derive h5_root from pt_files root (replace pt_files with patches)
        self.h5_root = [r.replace("/pt_files", "/patches").replace("\\pt_files", "\\patches") for r in self.root]
        
        self.csv_file = csv_file
        self.data = pd.read_csv(csv_file)
        
        # Determine split type
        if "fold" in self.data.columns:
            self.split = "fixed"
            self.num_folds = 1
        else:
            self.split = "5foldcv"
            self.num_folds = 5
        
        # Convert "label" column to discrete values
        self.data["label"] = pd.Categorical(self.data["label"])
        self.classes = self.data["label"].cat.categories.tolist()
        self.data["label"] = self.data["label"].cat.codes
        
        # Get number of classes
        self.num_classes = len(self.data["label"].unique())
        
        # Get the dimension of WSI features from first available slide
        self.n_features = None
        for root in self.root:
            pt_path = os.path.join(root, self.feature, str(self.data["slide"].values[0]) + ".pt")
            if os.path.exists(pt_path):
                self.n_features = torch.load(pt_path).shape[-1]
                break
        
        if self.n_features is None:
            raise ValueError(f"Could not find feature files in any of the roots: {self.root}")
        
        # Build case list
        self.cases = []
        for idx in range(len(self.data)):
            case = self.data.iloc[idx, :].values.tolist()[:3]
            self.cases.append(case)
        
        print("[dataset] GigaPath dataset from %s" % (self.csv_file))
        print("[dataset] number of cases=%d" % (len(self.cases)))
        print("[dataset] number of classes=%d" % (self.num_classes))
        print("[dataset] number of features=%d" % self.n_features)
        print("[dataset] h5_root(s)=%s" % self.h5_root)
        
        # Setup fold splits
        if self.split == "5foldcv":
            self.train = []
            self.test = []
            for fold in range(5):
                split = self.data["fold{}".format(fold + 1)].values.tolist()
                train_split = [i for i, x in enumerate(split) if x == "train"]
                test_split = [i for i, x in enumerate(split) if x == "test"]
                self.train.append(train_split)
                self.test.append(test_split)
                print("[dataset] fold %d, training split: %d, test split: %d" % (fold, len(train_split), len(test_split)))
        else:
            split = self.data["fold"].values.tolist()
            self.train = [i for i, x in enumerate(split) if x == "train"]
            self.val = [i for i, x in enumerate(split) if x == "val"]
            self.test = [i for i, x in enumerate(split) if x == "test"]
            print("[dataset] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))

    def get_fold(self, fold=0):
        if self.split == "fixed":
            assert fold == 0, "fold should be 0"
            print("[fetch *] training split: {}, validation split: {}, test split: {}".format(len(self.train), len(self.val), len(self.test)))
            return self.train, self.val, self.test
        elif self.split == "5foldcv":
            assert 0 <= fold <= 4, "fold should be in 0 ~ 4"
            print("[fetch *] fold %d, training split: %d, test split: %d" % (fold, len(self.train[fold]), len(self.test[fold])))
            return self.train[fold], self.test[fold]

    def __getitem__(self, index):
        case = self.cases[index]
        ID, Slide_name, Label = case
        
        slides = []
        coords_list = []
        
        for slide_part in str(Slide_name).split(";"):
            # Try to find the slide in any of the roots
            for root, h5_root in zip(self.root, self.h5_root):
                pt_path = os.path.join(root, self.feature, slide_part + ".pt")
                h5_path = os.path.join(h5_root, slide_part + ".h5")  # No feature subfolder for h5
                
                if os.path.exists(pt_path):
                    # Load features
                    slide_features = torch.load(pt_path)
                    slides.append(slide_features)
                    
                    # Load coordinates from h5 file
                    if os.path.exists(h5_path):
                        with h5py.File(h5_path, 'r') as f:
                            coords = torch.tensor(f['coords'][:], dtype=torch.float32)
                        coords_list.append(coords)
                    else:
                        # Create dummy coordinates when h5 file not found
                        num_patches = slide_features.shape[0]
                        dummy_coords = torch.zeros((num_patches, 2), dtype=torch.float32)
                        coords_list.append(dummy_coords)
                        print(f"[WARNING] Coordinates file not found: {h5_path}. Using dummy coordinates (zeros) for {num_patches} patches.")
                    break
        
        if len(slides) == 0:
            raise ValueError(f"Could not find slide {Slide_name} in any root")
        
        # Concatenate all slide parts
        Slide = torch.cat(slides, dim=0)
        Coords = torch.cat(coords_list, dim=0)
        
        # Ensure tensor types
        if not isinstance(Slide, torch.Tensor):
            raise ValueError("Slide is not a tensor")
        
        Label = torch.tensor(Label, dtype=torch.int64)
        
        return ID, Slide, Coords, Label

    def __len__(self):
        return len(self.cases)
