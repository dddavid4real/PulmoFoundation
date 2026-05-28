import os
import time
import re
from datetime import datetime

from datasets.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed, Survival_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
import torch
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset


def find_latest_checkpoint_dir(root_dir, model_name):
    """
    Find the latest checkpoint directory for a given model name.
    Directory format: [model_name]-[YYYY-MM-DD]-[HH-MM-SS]

    Args:
        root_dir: Root directory containing checkpoint folders
        model_name: Model name (e.g. 'AttMIL-PulmoFoundation-E2')

    Returns:
        Path to the latest checkpoint directory, or None if not found
    """
    print(f"[DEBUG] Searching in root_dir: {root_dir}")
    print(f"[DEBUG] Target model_name: {model_name}")

    if not os.path.isdir(root_dir):
        print(f"[ERROR] Provided root_dir is not a directory: {root_dir}")
        return None

    matching_dirs = []

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith(f"[{model_name}]-"):
            print(f"[DEBUG] Matched folder: {item}")
            matching_dirs.append(item)
        else:
            print(f"[DEBUG] Skipped item: {item}")

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
                parsed = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG] Parsed datetime for {dir_name}: {parsed}")
                return parsed
            else:
                print(f"[DEBUG] Regex did not match folder name: {dir_name}")
        except Exception as e:
            print(f"[ERROR] Failed to parse datetime for {dir_name}: {e}")
        return datetime(1900, 1, 1)

    matching_dirs.sort(key=parse_dir_datetime, reverse=True)
    latest_dir = matching_dirs[0]
    latest_dir_path = os.path.join(root_dir, latest_dir)

    print(f"[INFO] Latest checkpoint directory selected: {latest_dir_path}")
    return latest_dir_path

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    
    meter = Survival_Meter()
    
    # create results directory
    if args.evaluate:
        results_dir = args.resume  # Start with root directory
    else:
        results_dir = "./results/{modal}/{dataset}/[{model}-{feature}]-[{time}]".format(
            modal=args.modal,
            dataset=args.csv_file.split('/')[-1].split('.')[0],
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S")
        )
    print("[log dir] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # define dataset - use different dataset class for Gigapath (needs coordinates)
    if args.model == "Gigapath":
        from datasets.TCGA_Survival_Gigapath import TCGA_Survival_Gigapath
        dataset = TCGA_Survival_Gigapath(csv_file=args.csv_file, feature_path=args.feature_path, modal=args.modal, study=args.study, feature=args.feature)
    else:
        dataset = TCGA_Survival(csv_file=args.csv_file, feature_path=args.feature_path, modal=args.modal, study=args.study, feature=args.feature)
    args.num_classes = 4
    
    #* Automatically Load Feature Size
    args.data = pd.read_csv(args.csv_file)
    loaded = False
    root = args.feature_path
    for wsi_entry in args.data["WSI"]:
        # Handle multiple .pt files per entry
        pt_files = str(wsi_entry).split(";")
        first_pt = pt_files[0].strip()
        feature_file_path = os.path.join(root, args.feature, first_pt)
        print("Checking:", feature_file_path)

        if os.path.exists(feature_file_path):
            try:
                args.n_features = torch.load(feature_file_path, map_location="cpu").shape[-1]
                loaded = True
                break
            except Exception as e:
                print(f"Error loading feature file {feature_file_path}: {e}")

    if not loaded:
        raise FileNotFoundError("Could not find any valid feature file to determine n_features.")

    # get split
    train_split, val_split, test_split = dataset.get_split()
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split+test_split))
    test_subset = Subset(dataset, val_split+test_split)
    
    if args.evaluate:
        checkpoint_key = f"{args.model}-{args.feature}"
        results_dir_actual = find_latest_checkpoint_dir(results_dir, checkpoint_key)
        if results_dir_actual:
            results_dir = results_dir_actual
            print(f"[Evaluation] Using checkpoint directory: {results_dir}")
        else:
            print(f"[ERROR] Could not find checkpoint directory for {checkpoint_key}")
            return
    
    # build model, criterion, optimizer, schedular
    #################################################
    if args.model == "AttMIL":
        from models.AttMIL.network import DAttention
        from models.AttMIL.engine import Engine
        model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        engine = Engine(args, results_dir)
    elif args.model == "Gigapath":
        from models.Gigapath.network import GigapathClassifier
        from models.Gigapath.engine import Engine
        model = GigapathClassifier(n_classes=args.num_classes, n_features=args.n_features, freeze_encoder=False)
        engine = Engine(args, results_dir)
    elif args.model == "CHIEF":
        from models.CHIEF.network import ChiefClassifier
        from models.CHIEF.engine import Engine
        model = ChiefClassifier(n_classes=args.num_classes, n_features=args.n_features, freeze_encoder=False)
        engine = Engine(args, results_dir)
    else:
        raise NotImplementedError("model [{}] is not implemented".format(args.model))
    
    print('[model] trained model: ', args.model)
    criterion = define_loss(args)
    print('[model] loss function: ', args.loss)
    optimizer = define_optimizer(args, model)
    print('[model] optimizer: ', args.optimizer)
    scheduler = define_scheduler(args, optimizer)
    print('[model] scheduler: ', args.scheduler)
    
    result = engine.learning_with_bootstrap(model, train_loader, val_loader, test_subset, criterion, optimizer, scheduler)
    
    if result is None or any(x is None for x in result[:3]):
        print("[ERROR] Training/Evaluation returned None. Check logs for issues.")
        return
    
    mean_c_index, ci_lower, ci_upper, bootstrap_samples = result
    
    cindex_summary = f"{mean_c_index:.4f}({ci_lower:.4f},{ci_upper:.4f})"
    print(f"\n[Summary] C-Index: {cindex_summary}")
    
    # Update meter with results and bootstrap samples, then save
    meter.update(args, mean_c_index, ci_lower, ci_upper, bootstrap_samples)
    
    if args.evaluate:
        csv_result_path = os.path.join(results_dir, f"{args.study}_result.csv")
        print(f"[Evaluation] Saving results to: {csv_result_path}")
    else:
        csv_result_path = os.path.join(results_dir, "result.csv")
        print(f"[Training] Saving results to: {csv_result_path}")
    
    meter.save(csv_result_path)

if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
