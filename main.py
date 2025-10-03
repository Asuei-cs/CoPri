import os
from args import *
import numpy as np
import torch
from logger import init_logger
from utils import (
    set_seed, 
    EarlyStopping,
    check_path,
    get_model, 
    get_trainer,
    ProgramTimer,
    update_namespace_from_yaml,
    generate_filename
)
from datasets import data_partition, init_dataset, data_partition_leave_one_out  # type:ignore
import optuna
from trainers import *
import copy
from optuna.trial import TrialState


global data_loader # dataloader
global origin_data # origin_data  before bulid the dataloader 

data_loader = {}
origin_data = {}

@logger.catch


def save_domain_embeddings(args,model, path_prefix="final"):
    src_ids = torch.arange(args.itemnum1 + 1, model.item_emb.num_embeddings).to(args.device)
    tgt_ids = torch.arange(1, args.itemnum1 + 1).to(model.dev)
    src_emb = model.item_emb(src_ids).detach().cpu()
    tgt_emb = model.item_emb(tgt_ids).detach().cpu()
    torch.save({"src": src_emb, "tgt": tgt_emb}, f"{args.log_dir}/{path_prefix}_domain_embeddings.pt")

def train_model(args):

    set_seed(args.seed)
    
    init_logger(args)
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    if args.model == 'MGCL' and args.stage == "valid":
        model =  get_model(args.model)(copy.deepcopy(origin_data["user_train1_stage1"]), copy.deepcopy(origin_data["user_train2_stage1"]), args=args).to(args.device)
    elif args.model == 'MGCL' and args.stage == "test": 
        model =  get_model(args.model)(copy.deepcopy(origin_data["user_train1_stage2"]),copy.deepcopy(origin_data["user_train2_stage2"]), args=args).to(args.device)
    else:
        model = get_model(args.model)(args=args).to(args.device)
    
    trainer = get_trainer(args.trainer)(model,data_loader,args=args)

    # save_domain_embeddings(args,model, path_prefix="initial")
    
    if args.do_eval:
        if args.state_dict_path != None:
            trainer.load(args.state_dict_path)
            logger.info(f'Load model from {args.state_dict_path} for eval!')
            
        scores = trainer.eval(0)
        # save_domain_embeddings(args, trainer.model, path_prefix="final")

    else:
        if args.state_dict_path != None:
            try:
                logger.info(f'Load model from pre-model state dict: {args.state_dict_path} for Finetune!')
                trainer.load(args.state_dict_path)
            except Exception as e:
                logger.error(e)
                exit()
                
        # save model args
        args.checkpoint_path = generate_filename(args.log_dir, args)

        logger.info(f'the model save to: {args.checkpoint_path}')
        
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.stopping_step, verbose=True)
        with ProgramTimer():
            for epoch in range(1,args.num_epochs+1):
                trainer.train(epoch)
                
                if (epoch) % args.eval_freq  == 0:
                    scores = trainer.eval(epoch)
                    early_stopping(np.array([scores[args.valid_metric],]), trainer.model, epoch)
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        break
                    # break

            # load the best model to evaluate
            logger.info("load the best model to evaluate from {}".format(args.checkpoint_path))
            trainer.model.load_state_dict(torch.load(args.checkpoint_path))

            # save_domain_embeddings(args, trainer.model, path_prefix="final")

            scores = trainer.eval(0)
            logger.info(f"log file saved at: {args.file_log_dir}")

        return scores[args.valid_metric]


def objective(trial, args):
    ## 调优参数
    # optuna_args = argparse.Namespace(
    #     hidden_units=trial.suggest_categorical('hidden_units', [64, 128, 256]),
    #     num_layers=trial.suggest_categorical('num_layers', [1, 2, 3]),
    #     dropout_rate=trial.suggest_categorical('dropout_rate', [0,0.1,0.2,0.3, 0.4, 0.5]),
    #     l2_emb = trial.suggest_categorical('l2_emb', [0, 1e-3, 1e-2]),
    # )
    
    retries = 0
    
    while retries < args.max_retries:
    
        optuna_args = argparse.Namespace()
        # 从 opt_params 参数中拿到需要调整的参数。
        for param, values in args.opt_params.items():
            setattr(optuna_args, param, trial.suggest_categorical(param, values))
        
        args.trial_id = trial.number # 从optuna中拿到id
        
        # When you finish sampling, check whether there are trials with same values.
        #Modify the tuple of trial status, adding the running trial status to avoid sampling the same hyperparameter combination        
        sampled_trials = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,TrialState.RUNNING))
        found_duplicate = False
        for t in sampled_trials[::-1]:
            if t.number == trial.number:
                continue
            if trial.params == t.params:
                retries += 1
                found_duplicate = True
                print(f"重复的参数组合: {t.params}，正在重试（{retries}/{args.max_retries}）")
                break
        if not found_duplicate:
            break
            

    if retries == args.max_retries:
        print(f"超过最大重试次数 ({args.max_retries})，终止试验。")
        raise optuna.TrialPruned()

    args_dict = vars(args)
    args_dict.update(vars(optuna_args))
    args = argparse.Namespace(**args_dict)

    best_score = train_model(args)

    return best_score

def run_study(study_name,n_trials, gpu_id):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        study = optuna.create_study(study_name=study_name, 
                                    direction='maximize', 
                                    storage="sqlite:///hys4Rec_25_01_03.sqlite3",
                                    load_if_exists=True,
                                    sampler=optuna.samplers.GridSampler(args.opt_params)
                                    )  
        study.optimize(lambda trial: objective(trial, args), n_trials = n_trials) # type: ignore


if __name__ == '__main__':
    args , unknown_args = parse_args()

    update_namespace_from_yaml(args,yaml_path=f'./config/{args.model.lower()}.yaml',unknown_args=unknown_args)    


    args.log_dir = './log/'+args.target_domain + '_' + args.source_domain+'/' + args.model + '/' + args.study_tag + '/' + args.stage
    check_path(args.log_dir)

    
    # LOAD THE DATASET 
    if args.leave_one_out:
        origin_data = data_partition_leave_one_out(args.parent_dir, args.target_domain, args.source_domain)

    else:   
        # origin_data = data_partition(args.parent_dir, args.target_domain, args.source_domain)
        origin_data = data_partition(args.parent_dir, args.target_domain, args.source_domain,args.datasets)

     
    # [user_train, user_valid, user_test, usernum, itemnum] = dataset

    data_loader = init_dataset(origin_data, args)
    

    if args.optimize:
        study_name = args.model + "_" + args.target_domain + "_" + args.study_tag
        
        
        run_study(study_name=study_name,n_trials=args.n_trials,gpu_id=args.gpu_id)
    else:
        train_model(args)

    logger.info("Done") 

    

    
    
    
    
    

    
        