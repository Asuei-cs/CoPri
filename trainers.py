import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import *
from loguru import logger


class Trainer:
    def __init__(self, model, dataloaders, args):

        self.args = args
        self.model = model.to(args.device)


        self.dataloaders = dataloaders

        # self.data_name = self.args.data_name
        betas = (0.9, 0.98)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        logger.info(model)
        logger.info(f"Total Parameters:{sum([p.nelement() for p in self.model.parameters()])}")
        
        self.bce_criterion = nn.BCEWithLogitsLoss() 
        
        
    def train(self, epoch):
        self.iteration(epoch, self.dataloaders)
           
    def eval(self, epoch):
        return self.iteration(epoch, self.dataloaders, train=False)


    def iteration(self, epoch, *dataloader, train=True):
        raise NotImplementedError
    


    def get_full_sort_score(self, epoch, pred_list, truth_list):
        ks = self.args.ks
        sample_num = pred_list.shape[0]
         # 初始化
        total_metrics = {f'Recall@{k}': 0.0 for k in ks}
        total_metrics.update({f'NDCG@{k}': 0.0 for k in ks})

        for pred, truth in zip(pred_list, truth_list):
            # print(pred)
            # print(truth)
            for k in ks:
                recall = recall_at_k(pred, truth, k)
                ndcg = ndcg_at_k(pred, truth, k)
            
                total_metrics[f'Recall@{k}'] += recall                
                total_metrics[f'NDCG@{k}'] += ndcg

        
        for k in ks:
            total_metrics[f'Recall@{k}'] /= sample_num
            total_metrics[f'NDCG@{k}'] /= sample_num


        if epoch == 0:
            headers,values = dict2output(total_metrics)
            logger.info(headers)
            logger.info(values)            
        else:
            logger.info('epoch:{}'.format(epoch) +'    '+dict2str(total_metrics)) 
        
        return total_metrics

    def get_batch_sort_score(self, ks, total_metrics, batch_pred_list, batch_truth_list):
        
        cur_metrics = {f'Recall@{k}': [] for k in self.args.ks}
        cur_metrics.update({f'NDCG@{k}': [] for k in self.args.ks})

        for pred, truth in zip(batch_pred_list, batch_truth_list):
            for k in ks:
                recall = recall_at_k(pred, truth, k)
                ndcg = ndcg_at_k(pred, truth, k)
            
                total_metrics[f'Recall@{k}'] += recall                
                total_metrics[f'NDCG@{k}'] += ndcg
                cur_metrics[f'Recall@{k}'].append(recall)
                cur_metrics[f'NDCG@{k}'].append(ndcg)
                    
        return cur_metrics

    def toString(self, epoch, total_metrics):
        
        if epoch == 0:
            headers,values = dict2output(total_metrics)
            logger.info(headers)
            logger.info(values)            
        else:
            logger.info('epoch:{}'.format(epoch) +'    '+dict2str(total_metrics))
            
        return total_metrics



    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.args.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))


class CDSRTrainer(Trainer):
    def __init__(self, model, dataloaders, args):
        super().__init__(model, dataloaders, args=args)
        

    def iteration(self, epoch, dataloaders, train=True):
        
        str_code = "train" if train else "eval"

        if train:
            
            self.model.train()
            rec_data_iter_target = tqdm(enumerate(dataloaders["target_loader"]),
                    desc="Recommendation EP_%s_%s:%d" % (str_code, "target", epoch),
                    total=len(dataloaders["target_loader"]),
                    leave=False, 
                    ncols=100,
                    bar_format="{l_bar}{r_bar}")
            
            
            rec_data_iter_overlap = tqdm(enumerate(dataloaders["overlap_loader"]),
                desc="Recommendation EP_%s_%s:%d" % (str_code,"overlap",epoch),
                total=len(dataloaders["overlap_loader"]),
                leave=False, 
                ncols=100,
                bar_format="{l_bar}{r_bar}")
            
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            
            
            for i,batch in rec_data_iter_target:
                batch = tuple(np.array(t) for t in batch)
                loss = self.model.calculate_loss(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
            post_fix = {
                "epoch": epoch,
                "load_data":"target",
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss /(len(rec_data_iter_target)+len(rec_data_iter_overlap))),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }   
            
            if epoch  % self.args.log_freq == 0:
                logger.info(dict2str(post_fix))
            

            for i, batch in rec_data_iter_overlap:
                batch = tuple(np.array(t) for t in batch)
                loss = self.model.calculate_loss(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
            post_fix = {
                "epoch": epoch,
                "load_data":"overlap",
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / (len(rec_data_iter_target)+len(rec_data_iter_overlap))),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }   
            
            if  epoch % self.args.log_freq == 0:
                logger.info(dict2str(post_fix))            
            
            
        else: # eval default to full predict  we do not to impliment the sample predict
            self.model.eval()
         
            rec_data_iter_eval = tqdm(enumerate(dataloaders["eva_loader"]),
                    desc="Recommendation EP_%s:%d" % (str_code, epoch),
                    total=len(dataloaders["eva_loader"]),
                    leave=False, 
                    ncols=100,
                    bar_format="{l_bar}{r_bar}")   
            

            total_metrics = {f'Recall@{k}': 0.0 for k in self.args.ks}
            total_metrics.update({f'NDCG@{k}': 0.0 for k in self.args.ks})
            # res_data = {
            #     "user_id": [],
            #     "target_train_seq_len": [],
            #     "source_train_seq_len":[],
            # }
            # res_data.update({f'Recall@{k}': [] for k in self.args.ks})
            # res_data.update({f'NDCG@{k}': [] for k in self.args.ks})
            
            cnt_total_items = 0
            truth_list = []
            with torch.no_grad():
                for i, batch in rec_data_iter_eval:
                    # batch = tuple(np.array(t) for t in batch)
                    rating_pred = self.model.predict_full(batch)
                
                    if not self.args.sample_eval:
                        preds = torch.argsort(rating_pred, dim=1, descending=True) + 1
                    else:
                        try:

                            sampled_items = torch.tensor(np.array(batch[-2]), dtype=torch.long).to(self.args.device)
                            

                            preds = torch.argsort(rating_pred, dim=1, descending=True)

                            preds = torch.gather(sampled_items, 1, preds)
                            
                        except Exception as e:
                            logger.error(f"Error in sample eval: {str(e)}")
                            logger.error(f"sampled_items shape: {sampled_items.shape}")
                            logger.error(f"rating_pred shape: {rating_pred.shape}")
                            logger.error(f"min/max sampled_items: {sampled_items.min()}, {sampled_items.max()}")
                            raise e

                    cur_metrics = self.get_batch_sort_score(
                        self.args.ks, 
                        total_metrics, 
                        preds.cpu().numpy(),
                        batch[-1]
                    )
                    truth_list.extend(batch[-1])
                    # res_data["user_id"].extend(list(batch[0]))
                    # res_data["target_train_seq_len"].extend(list(np.count_nonzero(batch[1],axis=1)))
                    # res_data["source_train_seq_len"].extend(list(np.count_nonzero(batch[2],axis=1)))
                   
                    # for k in self.args.ks:
                    #     res_data[f'Recall@{k}'].extend(cur_metrics[f'Recall@{k}'])
                    #     res_data[f'NDCG@{k}'].extend(cur_metrics[f'NDCG@{k}'])
                                  
                 
            cnt_total_items = sum(len(items) for items in truth_list)

            logger.info(f'本次评估的物品数量有{cnt_total_items}个')
            valid_user_num = len(dataloaders["eva_loader"].dataset)
            logger.info(f'current type user nums is {valid_user_num}')  
            
            for k in self.args.ks:
                total_metrics[f'Recall@{k}'] /= valid_user_num
                total_metrics[f'NDCG@{k}'] /= valid_user_num

            
            return self.toString(epoch, total_metrics)
    
    
class OverlapTrainer(Trainer):
    def __init__(self, model, dataloaders, args):
        super().__init__(model, dataloaders , args=args)

    def iteration(self, epoch, dataloaders, train=True):
        
        str_code = "train" if train else "eval"

        if train:
            self.model.train()
            
            rec_data_iter_overlap = tqdm(enumerate(dataloaders["overlap_loader"]),
                desc="Recommendation EP_%s_%s:%d" % (str_code,"overlap",epoch),
                total=len(dataloaders["overlap_loader"]),
                leave=False, 
                ncols=100,
                bar_format="{l_bar}{r_bar}")
                 
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
           
            for i, batch in rec_data_iter_overlap:
                batch = tuple(np.array(t) for t in batch)
                loss = self.model.calculate_loss(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
            post_fix = {
                "epoch": epoch,
                "load_data":"overlap",
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss /len(rec_data_iter_overlap)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }   
            
            if  epoch % self.args.log_freq == 0:
                logger.info(dict2str(post_fix))

            
        else: # eval default to full predict  we do not to impliment the sample predict
            self.model.eval()
                
            rec_data_iter_eval = tqdm(enumerate(dataloaders["eva_loader"]),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloaders["eva_loader"]),
                leave=False, 
                ncols=100,
                bar_format="{l_bar}{r_bar}")   
            

            total_metrics = {f'Recall@{k}': 0.0 for k in self.args.ks}
            total_metrics.update({f'NDCG@{k}': 0.0 for k in self.args.ks})
            if self.args.is_save_user_rate:
                res_data = {
                    "user_id": [],
                    "target_train_seq_len": [],
                    "source_train_seq_len":[],
                }
                res_data.update({f'Recall@{k}': [] for k in self.args.ks})
                res_data.update({f'NDCG@{k}': [] for k in self.args.ks})
            
            cnt_total_items = 0
            truth_list = []
            with torch.no_grad():
                for i, batch in rec_data_iter_eval:
                    u, seq, seq2, mask, item_idx, items,mixed_seq,mixed_time=batch
                    # batch = tuple(np.array(t) for t in batch)
                    rating_pred = self.model.predict_full(batch)
                    # user, seq, seq2, mask, items[:1],mixed_seq,mixed_time
                
                    if not self.args.sample_eval:
                        preds = torch.argsort(rating_pred, dim=1, descending=True) + 1
                    else:
                        try:
                            sampled_items = torch.tensor(np.array(item_idx), dtype=torch.long).to(self.args.device)

                            preds = torch.argsort(rating_pred, dim=1, descending=True)

                            preds = torch.gather(sampled_items, 1, preds)
                            
                        except Exception as e:
                            logger.error(f"Error in sample eval: {str(e)}")
                            logger.error(f"sampled_items shape: {sampled_items.shape}")
                            logger.error(f"rating_pred shape: {rating_pred.shape}")
                            logger.error(f"min/max sampled_items: {sampled_items.min()}, {sampled_items.max()}")
                            raise e

                    cur_metrics = self.get_batch_sort_score(
                        self.args.ks, 
                        total_metrics, 
                        preds.cpu().numpy(),
                        items
                    )
                    truth_list.extend(items)
                    if self.args.is_save_user_rate:
                        res_data["user_id"].extend(list(batch[0]))
                        res_data["target_train_seq_len"].extend(list(np.count_nonzero(batch[1],axis=1)))
                        res_data["source_train_seq_len"].extend(list(np.count_nonzero(batch[2],axis=1)))

                        for k in self.args.ks:
                            res_data[f'Recall@{k}'].extend(cur_metrics[f'Recall@{k}'])
                            res_data[f'NDCG@{k}'].extend(cur_metrics[f'NDCG@{k}'])

                 
            cnt_total_items = sum(len(items) for items in truth_list)

            valid_user_num = len(dataloaders["eva_loader"].dataset)


            
            for k in self.args.ks:
                total_metrics[f'Recall@{k}'] /= valid_user_num
                total_metrics[f'NDCG@{k}'] /= valid_user_num
            if self.args.is_save_user_rate:
                output_dir = "different_user_analysis"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                res_df = pd.DataFrame(res_data)
                res_df.to_csv(f"./{output_dir}/different_user_res_{self.args.model}_{self.args.target_domain}_k{self.args.ks[-1]}.csv",index=False)
            
            return self.toString(epoch, total_metrics)
    
    
class OverlapTrainer_mark_items(Trainer):
    def __init__(self, model, dataloaders, args):
        super().__init__(model, dataloaders , args=args)
        logger.info(f"We use the OverlapTrainer_mark_items ")

    def iteration(self, epoch, dataloaders, train=True):
        
        str_code = "train" if train else "eval"

        if train:
            self.model.train()
            
            rec_data_iter_overlap = tqdm(enumerate(dataloaders["overlap_loader"]),
                desc="Recommendation EP_%s_%s:%d" % (str_code,"overlap",epoch),
                total=len(dataloaders["overlap_loader"]),
                leave=False, 
                ncols=100,
                bar_format="{l_bar}{r_bar}")
                 
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
           
            for i, batch in rec_data_iter_overlap:
                batch = tuple(np.array(t) for t in batch)
                loss = self.model.calculate_loss(batch)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                
            post_fix = {
                "epoch": epoch,
                "load_data":"overlap",
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss /len(rec_data_iter_overlap)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
            }   
            
            if  epoch % self.args.log_freq == 0:
                logger.info(dict2str(post_fix))            
            
        else: # eval default to full predict  we do not to impliment the sample predict
            self.model.eval()
                
            rec_data_iter_eval = tqdm(enumerate(dataloaders["eva_loader"]),
                desc="Recommendation EP_%s:%d" % (str_code, epoch),
                total=len(dataloaders["eva_loader"]),
                leave=False, 
                ncols=100,
                bar_format="{l_bar}{r_bar}")   
            

            total_metrics = {f'Recall@{k}': 0.0 for k in self.args.ks}
            total_metrics.update({f'NDCG@{k}': 0.0 for k in self.args.ks})
            
            cnt_total_items = 0
            truth_list = []
            with torch.no_grad():
                for i, batch in rec_data_iter_eval:
                    # batch = tuple(np.array(t) for t in batch)        # u, seq, seq2, mask, item_idx, items = batch
                    # u, seq, seq2, mask, item_idx, items 
                    # u, seq, seq2, mask, sampled_items, items, mark_items = batch
                    # 0 1 2 3 4 5 6 
                    
                    rating_pred = self.model.predict_full(batch)
                    # user, seq, seq2, mask, 


                    original_rating_pred = rating_pred.clone()

                    original_preds = torch.argsort(original_rating_pred, dim=1, descending=True) + 1
                    original_topk = original_preds[:, :20].cpu().numpy()  # 取前20个推荐

                    mark_items = torch.tensor(np.array(batch[6]), dtype=torch.bool).to(self.args.device)
                    rating_pred = rating_pred.masked_fill(~mark_items, -np.inf)  # 使用反向掩码

                    masked_preds = torch.argsort(rating_pred, dim=1, descending=True) + 1
                    masked_topk = masked_preds[:, :20].cpu().numpy()  # 取前20个推荐


                    if not self.args.sample_eval:

                        preds = masked_preds
                    else:
                        try:

                            sampled_items = torch.tensor(np.array(batch[-2]), dtype=torch.long).to(self.args.device)
                            

                            preds = torch.argsort(rating_pred, dim=1, descending=True)
                            

                            preds = torch.gather(sampled_items, 1, preds)
                    
                            
                        except Exception as e:
                            logger.error(f"Error in sample eval: {str(e)}")
                            logger.error(f"sampled_items shape: {sampled_items.shape}")
                            logger.error(f"rating_pred shape: {rating_pred.shape}")
                            logger.error(f"min/max sampled_items: {sampled_items.min()}, {sampled_items.max()}")
                            raise e


                            
                    cur_metrics = self.get_batch_sort_score(
                        self.args.ks, 
                        total_metrics, 
                        preds.cpu().numpy(),
                        batch[5]  # batch[5]是真实的items
                    )
                    truth_list.extend(batch[5])
                    # res_data["user_id"].extend(list(batch[0]))
                    # res_data["target_train_seq_len"].extend(list(np.count_nonzero(batch[1],axis=1)))
                    # res_data["source_train_seq_len"].extend(list(np.count_nonzero(batch[2],axis=1)))
                   
                    # for k in self.args.ks:
                    #     res_data[f'Recall@{k}'].extend(cur_metrics[f'Recall@{k}'])
                    #     res_data[f'NDCG@{k}'].extend(cur_metrics[f'NDCG@{k}'])
                                  
                 
            cnt_total_items = sum(len(items) for items in truth_list)

            # 统计用户真实交互的物品数量有多少 truth_list 中的数量 
            logger.info(f'本次评估的物品数量有{cnt_total_items}个')
            valid_user_num = len(dataloaders["eva_loader"].dataset)
            logger.info(f'current type user nums is {valid_user_num}')  
            
            for k in self.args.ks:
                total_metrics[f'Recall@{k}'] /= valid_user_num
                total_metrics[f'NDCG@{k}'] /= valid_user_num
            
            # output_dir = "different_user_analysis_k3000_increment"
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # res_df = pd.DataFrame(res_data)
            # res_df.to_csv(f"./{output_dir}/different_user_res_{self.args.model}_{self.args.target_domain}_k{self.args.ks[-1]}.csv",index=False)
            

            detailed_users = min(3, len(batch[0]))
            for i in range(detailed_users):
                user_id = batch[0][i]
                truth_item = batch[5][i][0] if len(batch[5][i]) > 0 else "无"
                

                original_set = set(original_topk[i, :10])
                masked_set = set(masked_topk[i, :10])
                overlap = len(original_set.intersection(masked_set))
                overlap_ratio = overlap / 10.0 * 100
                

                masked_items = np.where(batch[6][i] == 0)[0] + 1
                

                

                if truth_item != "无":
                    truth_item = int(truth_item)
                    original_rank = np.where(original_preds[i].cpu().numpy() == truth_item)[0]
                    masked_rank = np.where(masked_preds[i].cpu().numpy() == truth_item)[0]
                    
                    original_position = original_rank[0] + 1 if len(original_rank) > 0 else "未出现"
                    masked_position = masked_rank[0] + 1 if len(masked_rank) > 0 else "未出现"
                    
                    # print(f"目标物品在掩码前的排名: {original_position}")
                    # print(f"目标物品在掩码后的排名: {masked_position}")
                
                print(f"Top-10列表重叠度: {overlap}个物品 ({overlap_ratio:.1f}%)")
                
                # 检查是否有历史交互物品出现在掩码前的Top-10中
                hist_in_original = [item for item in original_topk[i, :10] if item in masked_items]
                if hist_in_original:
                    print(f"警告：掩码前Top-10中有{len(hist_in_original)}个历史交互物品: {hist_in_original}")


            all_original_top10 = original_preds[:, :10].cpu().numpy()
            all_masked_top10 = masked_preds[:, :10].cpu().numpy()

            total_overlap = 0
            for i in range(len(batch[0])):
                overlap = len(set(all_original_top10[i]).intersection(set(all_masked_top10[i])))
                total_overlap += overlap

            avg_overlap = total_overlap / (len(batch[0]) * 10.0) * 100
            print(f"\n批次总体统计:")
            print(f"平均Top-10重叠率: {avg_overlap:.2f}%")

            return self.toString(epoch, total_metrics)