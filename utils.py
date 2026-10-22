import os
import torch
import numpy as np
import random
import copy
from tqdm import tqdm
from importlib import util as importlib_util,import_module
from datetime import datetime
from functools import wraps
from loguru import logger
import yaml
import argparse
import time

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_epoch = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if self.best_score is None:
                return TypeError
            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model, cur_epoch):
        # score NDCG@20
        
        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.best_epoch = cur_epoch
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}. best score is {self.best_score}. best epoch is {self.best_epoch}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = cur_epoch
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f})
            logger.info(f"Validation score increased. best score is {self.best_score}, best epoch is {self.best_epoch}. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def dict2output(result_dict):
    column_widths = {key: max(len(key), len(f"{value:.6f}")) for key, value in result_dict.items()}

    header = "\t".join(f"{key:<{column_widths[key]}}" for key in result_dict.keys())

    values = "\t".join(f"{result_dict[key]:<{column_widths[key]}.6f}" for key in result_dict.keys())

    return header, values
    
def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    return '    '.join([str(metric) + ' : ' + str(value) for metric, value in result_dict.items()])



def recalls_and_ndcgs_for_ks(scores, labels, ks):
    r'''
    Evaluate with recall and NDCG
    
    Require the tensor data input 

    '''
    
    metrics = {}

    scores = scores.cpu()
    labels = labels.cpu()

    answer_count = labels.sum(1)

    answer_count_float = answer_count.float()
    labels_float = labels.float()

    rank = (-scores).argsort(dim=1)
    cut = rank.to(labels.device)
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = (hits.sum(1) / answer_count_float).sum().item()

       position = torch.arange(2, 2+k).to(labels.device)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights).sum(1)
       idcg = torch.Tensor([weights[:min(n, k)].sum() for n in answer_count])
       ndcg = (dcg / idcg).sum().item()
       metrics['NDCG@%d' % k] = ndcg

    return metrics  

def calculate_metrics(pred, truth, ks):
    metrics = {}
    for k in ks:
        recall = recall_at_k(pred, truth, k)
        ndcg = ndcg_at_k(pred, truth, k)
        metrics[f'Recall@{k}'] = recall
        metrics[f'NDCG@{k}'] = ndcg
    return metrics


def dcg_at_k(r, k):
    """Compute DCG at rank k."""
    # r = np.asfarray(r)[:k]
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(pred_list, truth_list, k):
    """Compute NDCG at rank k for a single user using numpy arrays."""
    ideal_relevance = np.ones(min(len(truth_list), k))
    idcg = dcg_at_k(ideal_relevance, k)
    truth_list = np.array(truth_list)

    relevance = np.isin(pred_list[:k], truth_list).astype(int)
    dcg = dcg_at_k(relevance, k)

    return dcg / idcg if idcg > 0 else 0.


def recall_at_k(pred_list, truth_list, k):
    """Compute Recall at rank k for a single user using numpy arrays."""

    pred_list = pred_list[:k]

    hits = np.intersect1d(pred_list, truth_list).size

    recall = hits / len(truth_list) if len(truth_list) > 0 else 0

    
    return recall


def get_model(model_name):
    model_file_name = model_name.lower()
    module_path = f'models.{model_file_name}'

    if importlib_util.find_spec(module_path):
        model_module = import_module(module_path)
        model_class = getattr(model_module, model_name)
    else:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    
    return model_class

def get_dataset(dataset_name:str):
    module_path = 'datasets'
    try:

        dataset_module = import_module(module_path)
        

        dataset_class = getattr(dataset_module, dataset_name, None)
        

        if dataset_class is None:
            raise AttributeError(f"Dataset '{dataset_name}' not found in module '{module_path}'")
        

        return dataset_class
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{module_path}' not found")
    except AttributeError as e:
        raise e

def get_trainer(trainer_name:str):
    module_path = 'trainers'
    try:

        trainer_module = import_module(module_path)

        trainer_class = getattr(trainer_module, trainer_name, None)
        

        if trainer_class is None:
            raise AttributeError(f"Trainer '{trainer_name}' not found in module '{module_path}'")
        

        return trainer_class
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{module_path}' not found")
    except AttributeError as e:
        raise e
    


def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"Started execution of {func.__name__} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info(f"Finished execution of {func.__name__} at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Elapsed time: {str(elapsed_time).split('.')[0]} (hh:mm:ss)")
        return result
    return wrapper


class ProgramTimer:
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Program started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        logger.info(f"Program finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                    f"Total elapsed time: {str(elapsed_time).split('.')[0]} (hh:mm:ss)")


def update_namespace_from_yaml(namespace, yaml_path, unknown_args={}):
    """
    Updates the argparse.Namespace object with values from a YAML file and unknown command line arguments.
    Unknown arguments from command line have the highest priority.
    """

    def convert_value(value):

        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        elif isinstance(value, (int, float)) and value == int(value) and value > 0:
            value = int(value)  # Ensure positive integer
        elif isinstance(value, list):
            value = [convert_value(v) for v in value]
        elif isinstance(value, dict):
            value = {k: convert_value(v) for k, v in value.items()}
        return value

    with open(yaml_path, 'r') as file:
        yaml_config = yaml.safe_load(file)

    unknown_dict = {}
    for arg in unknown_args:
        if '=' in arg:
            key, value = arg.split('=')
            key = key.lstrip('--')
            unknown_dict[key] = convert_value(value)
            

    for key, value in unknown_dict.items():
       
        if key in yaml_config.keys():
            yaml_config[key] = unknown_dict[key]
        else:
            raise ValueError(f"Unknown argument '{key}' not found in the config file")

    for key, value in yaml_config.items():
            # if not hasattr(namespace, key):
        setattr(namespace, key, convert_value(value))

    
def generate_filename(logdir, args):
    """
    generate the dir with the model state dict path 
    """

    fname = f'[{args.stage}]_[{args.model}]_[trial={args.trial_id}]_{args.current_time}_{args.md5}'
    if args.optimize:

        for k in args.opt_params.keys():
            fname += f'.{k}={getattr(args, k)}'
    
    fname += f'.pth'    
    
    return os.path.join(logdir,fname)     

if __name__ == "__main__":

    from args import *
    args,unknown_args = parse_args()
    update_namespace_from_yaml(args,yaml_path=f'./config/{args.model.lower()}.yaml',unknown_args=unknown_args)    
    print(args)
    args.usernum = 1000
    args.usernum1 = 100
    args.usernum2 = 900
    args.itemnum1 = 10
    args.itemnum2 = 20
    args.itemnum = 10 +20

    
    dataset = get_dataset("CDSRDataset")(user_train1=[],
                            user_train2=[],
                            user_eva=[],
                            time1 =[],
                            time2=[],
                            args=args,
                            user_eva2=[])
    print(dataset)
    