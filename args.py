import argparse
from email.policy import default


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

def str_to_list(input_str):
    try:
        import ast
        return ast.literal_eval(input_str)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid list format")


def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Domain Sequential Recommendation with Item Text Descriptions for Partial-Overlapping Users')
    # Data parameters
    # parser.add_argument('--target_domain',"-td", default='computers', type=str, help='target dataset name')
    # parser.add_argument('--source_domain',"-sd", default='appliances', type=str, help='source dataset name')
    # parser.add_argument('--parent_dir', default='Data/Rees46/', type=str, help='path to the parent directory of the dataset')
    # parser.add_argument('--datasets',default='Rees46',type=str)
    parser.add_argument('--target_domain',"-td", default='Cell', type=str, help='target dataset name')
    parser.add_argument('--source_domain',"-sd", default='Elec', type=str, help='source dataset name')
    parser.add_argument('--parent_dir', default='Data/Amazon/', type=str, help='path to the parent directory of the dataset')
    parser.add_argument('--datasets',default='Amazon',type=str)

    # Training parameters
    parser.add_argument('--seed', type=int, default=2025,
                        help='random seed for random, numpy and torch (default: 2020).')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout_rate (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='batch size for eval')
    
    parser.add_argument('--num_epochs', default=500, type=int)
    parser.add_argument('--device', default='cuda:3', type=str,choices=["cuda","cpu","cuda:0","cuda:1","cuda:2","cuda:3"],)
    
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--eval_freq", type=int, default=1, help="eval freq for eval.")
    parser.add_argument('--do_eval', default=False, type=str2bool)
    
    parser.add_argument('--state_dict_path', type=str)
    
    parser.add_argument('--stage', default='valid', type=str,choices=["valid","test"])
    
    parser.add_argument('--model', default='CoPri', type=str,choices=["CoPri","GRU4Rec","GCSAN","SASRec","FMLP","CDSASRec","TJAPL","MGCL","ABXI","HGNCDR"])
    
    parser.add_argument('--trainer', default='OverlapTrainer', type=str,help="to choice the corresponding Trainer for which type you want to train",choices=['CDSRTrainer','OverlapTrainer']) 

    parser.add_argument('--dataset', default='CDSRDataset_Increment', type=str,help="to choice the corresponding Dataset for which type you want to train",choices=['CDSRDataset','CDSRDataset_Increment',])

    parser.add_argument('--sample_eval', default=False, type=str2bool, help="if this key's value is true, we use the sample eval method, otherwise we use the all items eval method")

    parser.add_argument('--leave_one_out', default=False, type=str2bool, help="if this key's value is true, we use the leave one out method to partition the data, else we use the timestamp partition method")
    
    parser.add_argument('--eva_type',default='overlap', type=str,help="to choice the corresponding eval type for which type you want to eval",choices=["target_overlap","overlap"])
    
    parser.add_argument('--ks', default=[3,5,10], type=str_to_list)
    parser.add_argument('--stopping_step', default=20, type=int)
    parser.add_argument('--valid_metric',default="NDCG@5",type=str)
    
    parser.add_argument('--maxlen', default=50, type=int)

    parser.add_argument('--eval_only_first', default=True, type=str2bool, help="if this key's value is true, we only eval the first item for each user, else we eval all items for each user")

    # for hyps select 
    # NOTICE: if the stage is test , we can not use the optimize to select hyps,
    # in other word, we only select hyps in valid stage
    parser.add_argument('--optimize', type=str2bool, default=False, help="Use Optuna for hyperparameter optimization")
    parser.add_argument('--trial_id', type=int, default=-1, help="for optuna to optimize, if not use optuna ,the default to -1 ,else use the trial id when optimize ")
    parser.add_argument('--gpu_id', type=int, default=0, help="for optuna to optimize, the gpu_id is your device's gpus , you can set it to choice so that gpu will be used or not")
    parser.add_argument('--n_trials', type=int, default=10, help="for optuna to optimize, the total trials that you want, for each process, the trial_num is total_n_trials//process_nums")
    parser.add_argument('--max_retries', type=int, default=10, help="for optuna to optimize, the value is to avoid reselecting parameters too many times, which may cause the program to deadlock. ")
    parser.add_argument('--study_tag',type=str,default='test_myModel',help="for optuna to optimize, the tag is to distinguish the different study, if you want to run the same study, you can set the same tag")
    parser.add_argument('--is_save_user_rate', default=False, type=str2bool)
    return parser.parse_known_args()

if __name__ == "__main__":
    # Unit Test for data type
    args, unknown_args = parse_args()
    print(type(args.state_dict_path))
    if args.state_dict_path == None:
        print("YES")
    else :
        print("NO")
        