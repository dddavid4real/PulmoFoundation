import os
import time
import re
from datetime import datetime
import pickle

from datasets.Subtyping import Dataset_Subtyping
from utils.options import parse_args
from utils.util import set_seed, CV_Meter
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

from torch.utils.data import DataLoader, SubsetRandomSampler

def find_latest_checkpoint_dir(root_dir, model_name):
    """
    Find the latest checkpoint directory for a given model name.
    Directory format: [model_name]-[YYYY-MM-DD]-[HH-MM-SS]

    Args:
        root_dir: Root directory containing checkpoint folders
        model_name: Model name (e.g. 'chief', 'conch15', etc.)

    Returns:
        Path to the latest checkpoint directory, or None if not found
    """

    if not os.path.isdir(root_dir):
        print(f"[ERROR] Provided root_dir is not a directory: {root_dir}")
        return None

    matching_dirs = []

    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith(f"[{model_name}]-"):
            print(f"Matched folder: {item}")
            matching_dirs.append(item)
        else:
            print(f"Skipped item: {item}")

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
                print(f"Parsed datetime for {dir_name}: {parsed}")
                return parsed
            else:
                print(f"Regex did not match folder name: {dir_name}")
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
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        results_dir = "./results/results_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print("[log dir] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset - use different dataset class for Gigapath
    if args.model == "Gigapath":
        from datasets.Subtyping_Gigapath import Dataset_Subtyping_Gigapath
        dataset = Dataset_Subtyping_Gigapath(root=args.root, csv_file=args.csv_file, feature=args.feature)
    else:
        dataset = Dataset_Subtyping(root=args.root, csv_file=args.csv_file, feature=args.feature)
    # training and evaluation
    meter = CV_Meter(dataset.num_folds)
    args.num_classes = dataset.num_classes
    args.n_features = dataset.n_features
    args.num_folds = dataset.num_folds
    for fold in range(dataset.num_folds):
        splits = dataset.get_fold(fold)
        loaders = [DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(split)) for split in splits]
        # build model, criterion, optimizer, schedular
        #################################################
        if args.model == "ABMIL":
            from models.ABMIL.network import DAttention
            from models.ABMIL.engine import Engine

            model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
            engine = Engine(args, results_dir, fold)
        elif args.model == "Gigapath":
            from models.Gigapath.network import GigapathClassifier
            from models.Gigapath.engine import Engine
            
            model = GigapathClassifier(
                n_classes=args.num_classes, 
                n_features=args.n_features,
                freeze_encoder=False
            )
            engine = Engine(args, results_dir, fold)
        elif args.model == "CHIEF":
            from models.CHIEF.network import ChiefClassifier
            from models.CHIEF.engine import Engine
            
            model = ChiefClassifier(
                n_classes=args.num_classes,
                n_features=args.n_features,
                freeze_encoder=False
            )
            engine = Engine(args, results_dir, fold)
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)
        
        if args.num_folds > 1:
            if args.evaluate:
                test_scores, test_scores_bootstrapped, bootstrap_samples, predictions = engine.learning(
                    model, loaders, criterion, optimizer, scheduler
                )
                meter.updata(0, test_scores, test_scores_bootstrapped, bootstrap_samples)
                
                # Save predictions with ROC bootstrap data
                results_dir_actual = find_latest_checkpoint_dir(results_dir, args.feature)
                pred_filename = f"{args.study}_predictions.pkl"
                pred_path = os.path.join(results_dir_actual, pred_filename)
                with open(pred_path, 'wb') as f:
                    pickle.dump(predictions, f)
                print(f"Saved predictions to: {pred_path}")
            else:
                val_scores, best_epoch = engine.learning(model, loaders, criterion, optimizer, scheduler)
                meter.updata(best_epoch, val_scores)
        else:
            if args.evaluate:
                test_scores, test_scores_bootstrapped, bootstrap_samples, predictions = engine.learning(
                    model, loaders, criterion, optimizer, scheduler
                )
                meter.updata(0, test_scores, test_scores_bootstrapped, bootstrap_samples)
                
                # Save predictions
                results_dir_actual = find_latest_checkpoint_dir(results_dir, args.feature)
                pred_filename = f"{args.study}_predictions.pkl"
                pred_path = os.path.join(results_dir_actual, pred_filename)
                with open(pred_path, 'wb') as f:
                    pickle.dump(predictions, f)
                print(f"Saved predictions to: {pred_path}")
            else:
                val_scores, test_scores, best_epoch, bootstrap_samples, predictions = engine.learning(
                    model, loaders, criterion, optimizer, scheduler
                )
                meter.updata(best_epoch, val_scores, test_scores, bootstrap_samples)
                
                # Save predictions
                pred_filename = "predictions.pkl"
                pred_path = os.path.join(results_dir, pred_filename)
                with open(pred_path, 'wb') as f:
                    pickle.dump(predictions, f)
                print(f"Saved predictions to: {pred_path}")
    
    if not args.evaluate:
        meter.save(os.path.join(results_dir, "result.csv"))
    else:
        results_dir = find_latest_checkpoint_dir(results_dir, args.feature)
        print("[log dir] results directory: ", results_dir)
        # Save evaluation results using the same save method
        meter.save(os.path.join(results_dir, "{study}_result.csv".format(study=args.study)))


if __name__ == "__main__":

    args = parse_args()
    results = main(args)
    print("finished!")
