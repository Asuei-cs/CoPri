import os
import sys
import copy
import pandas as pd
import numpy as np
from numpy import dtype
from torch.utils.data import Dataset, DataLoader
from utils import set_seed, get_dataset
import random
from loguru import logger

def custom_collate_fn(batch):
    *other_elements, items = zip(*batch)

    return (*other_elements, list(items))

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def gen_neg_sample(df_data, sample_num=100):

    sr_user2items = df_data.groupby(['user_id']).item_id.unique()
    df_negative = pd.DataFrame({'user_id': df_data.user_id.unique()})

    sr_item2pop = df_data.item_id.value_counts(sort=True, ascending=False)
    arr_item = sr_item2pop.index.values
    arr_pop = sr_item2pop.values

    def get_negative_sample(pos):
        neg_idx = ~np.in1d(arr_item, pos)
        neg_item = arr_item[neg_idx]
        neg_pop = arr_pop[neg_idx]
        neg_pop = neg_pop / neg_pop.sum()

        return np.random.choice(neg_item, size=sample_num, replace=False, p=neg_pop)
    
    arr_sample = df_negative.user_id.apply(
        lambda x: get_negative_sample(sr_user2items[x])).values
    
    df_negative = pd.concat([df_negative, pd.DataFrame(arr_sample.tolist())], axis=1)


    return df_negative

def read_Rees46(fname, fname2):
    local_path = "Data/Rees46/"
    
    train_all = pd.read_csv(local_path + '/train_data.csv', header=0)
    valid_all = pd.read_csv(local_path + '/valid_data.csv', header=0)
    test_all = pd.read_csv(local_path + '/test_data.csv', header=0)


    for df in [train_all, valid_all, test_all]:
        df['timestamp'] = pd.to_datetime(df['event_time']).astype('int64') // 1000000000

    d1_train_stage1 = train_all.loc[train_all['domain'] == fname]
    d1_val_stage1 = valid_all.loc[valid_all['domain'] == fname]
    d1_train_stage2 = pd.concat([d1_train_stage1, d1_val_stage1])
    d1_test_stage2 = test_all.loc[test_all['domain'] == fname]

    d2_train_stage1 = train_all.loc[train_all['domain'] == fname2]
    d2_val_stage1 = valid_all.loc[valid_all['domain'] == fname2]
    d2_train_stage2 = pd.concat([d2_train_stage1, d2_val_stage1])
    d2_test_stage2 = test_all.loc[test_all['domain'] == fname2]

    overlap_users = train_all['user_id'].unique()
    d1_items = d1_train_stage2['item_id'].unique()
    d2_items = d2_train_stage2['item_id'].unique()

    def map_ids(dataframes, column, mapping):
        for df in dataframes:
            df.loc[:, column] = df[column].map(mapping)


    user_map = {old_id: new_id for new_id, old_id in enumerate(set(overlap_users), 1)}
    map_ids([d1_train_stage1, d1_val_stage1, d1_train_stage2, d1_test_stage2,
             d2_train_stage1, d2_val_stage1, d2_train_stage2, d2_test_stage2], 'user_id', user_map)

    d1_item_map = {old_id: new_id for new_id, old_id in enumerate(set(d1_items), 1)}
    map_ids([d1_train_stage1, d1_val_stage1, d1_train_stage2, d1_test_stage2], 'item_id', d1_item_map)

    d2_item_map = {old_id: new_id for new_id, old_id in enumerate(set(d2_items), len(d1_items) + 1)}
    map_ids([d2_train_stage1, d2_val_stage1, d2_train_stage2, d2_test_stage2], 'item_id', d2_item_map)

    return (d1_train_stage1,d1_val_stage1,d1_train_stage2,d1_test_stage2,
            d2_train_stage1,d2_val_stage1,d2_train_stage2,d2_test_stage2)

def read_Amazon(fname, fname2):
    local_path = "Data/Amazon"
    folder_path = fname + "_" + fname2

    d1_train_stage1 = pd.read_csv("%s/%s/%s_train_stage1.csv" % (local_path,folder_path, fname))
    d1_val_stage1 = pd.read_csv("%s/%s/%s_val_stage1.csv" % (local_path,folder_path, fname))
    d1_train_stage2 = pd.read_csv("%s/%s/%s_train_stage2.csv" % (local_path,folder_path, fname))
    d1_test_stage2 = pd.read_csv("%s/%s/%s_test_stage2.csv" % (local_path,folder_path, fname))

    d2_train_stage1 = pd.read_csv("%s/%s/%s_train_stage1.csv" % (local_path,folder_path, fname2))
    d2_val_stage1 = pd.read_csv("%s/%s/%s_val_stage1.csv" % (local_path,folder_path, fname2))
    d2_train_stage2 = pd.read_csv("%s/%s/%s_train_stage2.csv" % (local_path,folder_path, fname2))
    d2_test_stage2 = pd.read_csv("%s/%s/%s_test_stage2.csv" % (local_path,folder_path, fname2))

    return (d1_train_stage1, d1_val_stage1, d1_train_stage2, d1_test_stage2,
            d2_train_stage1, d2_val_stage1, d2_train_stage2, d2_test_stage2)

def data_partition(parent_dir,fname,fname2,datasets):
    folder_path = fname + "_" + fname2
    folder_path = parent_dir + "/" + folder_path

    if datasets=='Rees46':
        d1_train_stage1, d1_val_stage1, d1_train_stage2, d1_test_stage2,d2_train_stage1, d2_val_stage1, d2_train_stage2, d2_test_stage2=read_Rees46(fname,fname2)
    elif datasets=='Amazon':
        d1_train_stage1, d1_val_stage1, d1_train_stage2, d1_test_stage2, d2_train_stage1, d2_val_stage1, d2_train_stage2, d2_test_stage2 = read_Amazon(
        fname, fname2)

    d1_train_stage1.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d1_val_stage1.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d1_train_stage2.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d1_test_stage2.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    
    d2_train_stage1.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d2_val_stage1.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d2_train_stage2.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d2_test_stage2.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)

    user_train1_stage1= d1_train_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_valid1_stage1 = d1_val_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_train1_stage2 = d1_train_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    user_test1_stage2 = d1_test_stage2.groupby("user_id")["item_id"].apply(list).to_dict()  
    
    user_train2_stage1 = d2_train_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_valid2_stage1 = d2_val_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_train2_stage2 = d2_train_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    user_test2_stage2 = d2_test_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    
    usernum1 = int(d1_train_stage2.user_id.nunique())
    usernum2 = int(d2_train_stage2.user_id.nunique())
    
    itemnum1 = int(d1_train_stage2.item_id.nunique())
    itemnum2 = int(d2_train_stage2.item_id.nunique())
    
    usernum = len(set(list(user_train1_stage2.keys()) + list(user_train2_stage2.keys())))

    Time1_stage1 = d1_train_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time1_stage2 = d1_train_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage1 = d2_train_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage2 = d2_train_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    

    Time1_stage1_val = d1_val_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time1_stage2_test = d1_test_stage2.groupby("user_id")["timestamp"] .apply(list).to_dict()
    Time2_stage1_val = d2_val_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage2_test = d2_test_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    
    
    def merge_dict(dict1,dict2):
        for key in dict2:
            if key in dict1:
                dict1[key].extend(dict2[key])
            else:
                raise KeyError(key)
            
    merge_dict(Time1_stage1,Time1_stage1_val)
    merge_dict(Time1_stage2,Time1_stage2_test) 
    merge_dict(Time2_stage1,Time2_stage1_val)
    merge_dict(Time2_stage2,Time2_stage2_test)

    del Time1_stage1_val, Time1_stage2_test, Time2_stage1_val, Time2_stage2_test
    
    return_dict = {
        "user_train1_stage1": user_train1_stage1,
        "user_valid1_stage1": user_valid1_stage1,
        "user_train1_stage2": user_train1_stage2,
        "user_test1_stage2": user_test1_stage2,
        "usernum1": usernum1,
        "itemnum1": itemnum1,
        "user_train2_stage1": user_train2_stage1,
        "user_valid2_stage1": user_valid2_stage1,
        "user_train2_stage2": user_train2_stage2,
        "user_test2_stage2": user_test2_stage2,
        "usernum2": usernum2,
        "itemnum2": itemnum2,
        "Time1_stage1": Time1_stage1,
        "Time1_stage2": Time1_stage2,
        "Time2_stage1": Time2_stage1,
        "Time2_stage2": Time2_stage2,
        "usernum": usernum,

    }
    d1_neg_file = f"{folder_path}/{fname}_neg_samples.csv"
    if os.path.exists(d1_neg_file):
        d1_neg = pd.read_csv(d1_neg_file, header=None)
    else:
        d1_neg = gen_neg_sample(pd.concat([d1_train_stage2,d1_test_stage2]))
        d1_neg.to_csv(d1_neg_file, index=False, header=False)

    d1_neg_dict = dict(zip(d1_neg.iloc[:, 0], d1_neg.iloc[:, 1:].values.tolist()))
    return_dict["negatvie_items"] = d1_neg_dict

    return return_dict

def data_partition_leave_one_out(parent_dir, fname, fname2):


    folder_path = os.path.join(parent_dir, f"{fname}_{fname2}")

    d1_all = pd.concat([
        pd.read_csv(f"{folder_path}/{fname}_train_stage2.csv"),
        pd.read_csv(f"{folder_path}/{fname}_test_stage2.csv")
    ])

    d2_all = pd.concat([
        pd.read_csv(f"{folder_path}/{fname2}_train_stage2.csv"),
        pd.read_csv(f"{folder_path}/{fname2}_test_stage2.csv")
    ])


    d1_all.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    d2_all.sort_values(by=['timestamp'], kind='mergesort', ascending=True, inplace=True)
    

    d1_grouped = d1_all.groupby('user_id')
    d1_train_stage1 = []
    d1_val_stage1 = []
    d1_test_stage2 = []
    
    for _, group in d1_grouped:
        if len(group) > 3:  # Need at least 3 records
            d1_train_stage1.append(group.iloc[:-2])
            d1_val_stage1.append(group.iloc[-2:-1])
            d1_test_stage2.append(group.iloc[-1:])

    d1_train_stage1 = pd.concat(d1_train_stage1)
    d1_val_stage1 = pd.concat(d1_val_stage1)
    d1_test_stage2 = pd.concat(d1_test_stage2)
    

    d2_grouped = d2_all.groupby('user_id')
    d2_train_stage1 = []
    d2_val_stage1 = []
    d2_test_stage2 = []
    
    for _, group in d2_grouped:
        if len(group) > 3:
            d2_train_stage1.append(group.iloc[:-2])
            d2_val_stage1.append(group.iloc[-2:-1])
            d2_test_stage2.append(group.iloc[-1:])
    
    d2_train_stage1 = pd.concat(d2_train_stage1)
    d2_val_stage1 = pd.concat(d2_val_stage1)
    d2_test_stage2 = pd.concat(d2_test_stage2)
    

    d1_train_stage2 = pd.concat([d1_train_stage1, d1_val_stage1])
    d2_train_stage2 = pd.concat([d2_train_stage1, d2_val_stage1])
    

    user_train1_stage1 = d1_train_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_valid1_stage1 = d1_val_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_train1_stage2 = d1_train_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    user_test1_stage2 = d1_test_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    
    user_train2_stage1 = d2_train_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_valid2_stage1 = d2_val_stage1.groupby("user_id")["item_id"].apply(list).to_dict()
    user_train2_stage2 = d2_train_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    user_test2_stage2 = d2_test_stage2.groupby("user_id")["item_id"].apply(list).to_dict()
    

    usernum1 = d1_all.user_id.nunique()
    usernum2 = d2_all.user_id.nunique()
    itemnum1 = d1_all.item_id.nunique()
    itemnum2 = d2_all.item_id.nunique()
    usernum = len(set(list(d1_all.user_id.unique()) + list(d2_all.user_id.unique())))
    

    Time1_stage1 = d1_train_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time1_stage2 = d1_train_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage1 = d2_train_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage2 = d2_train_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    

    def merge_dict(dict1, dict2):
        for key in dict2:
            if key in dict1:
                dict1[key].extend(dict2[key])
            else:
                raise KeyError(f"key {key} not in dict1")
    
    Time1_stage1_val = d1_val_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time1_stage2_test = d1_test_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage1_val = d2_val_stage1.groupby("user_id")["timestamp"].apply(list).to_dict()
    Time2_stage2_test = d2_test_stage2.groupby("user_id")["timestamp"].apply(list).to_dict()
    
    merge_dict(Time1_stage1, Time1_stage1_val)
    merge_dict(Time1_stage2, Time1_stage2_test)
    merge_dict(Time2_stage1, Time2_stage1_val)
    merge_dict(Time2_stage2, Time2_stage2_test)
    return_dict = {
        "user_train1_stage1": user_train1_stage1,
        "user_valid1_stage1": user_valid1_stage1,
        "user_train1_stage2": user_train1_stage2,
        "user_test1_stage2": user_test1_stage2,
        "usernum1": usernum1,
        "itemnum1": itemnum1,
        "user_train2_stage1": user_train2_stage1,
        "user_valid2_stage1": user_valid2_stage1,
        "user_train2_stage2": user_train2_stage2,
        "user_test2_stage2": user_test2_stage2,
        "usernum2": usernum2,
        "itemnum2": itemnum2,
        "Time1_stage1": Time1_stage1,
        "Time1_stage2": Time1_stage2,
        "Time2_stage1": Time2_stage1,
        "Time2_stage2": Time2_stage2,
        "usernum": usernum
    }

    d1_neg_file = f"{folder_path}/{fname}_neg_samples.csv"
    if os.path.exists(d1_neg_file):
        d1_neg = pd.read_csv(d1_neg_file, header=None)
    else:
        logger.info(f"no find neg_file,get neg_file: {d1_neg_file}")
        d1_neg = gen_neg_sample(d1_all)
        d1_neg.to_csv(d1_neg_file, index=False, header=False)

    d1_neg_dict = dict(zip(d1_neg.iloc[:, 0], d1_neg.iloc[:, 1:].values.tolist()))
    return_dict["negatvie_items"] = d1_neg_dict
    
    return return_dict

class baseDataset(Dataset):
    def __init__(self, user_train1, user_train2,user_eva, time1, time2, args, user_type, data_type):
        self.user_train1 =  user_train1
        self.user_train2 = user_train2
        self.user_eva = user_eva
        self.time1 = time1
        self.time2 = time2
        self.usernum = args.usernum
        self.usernum1 = args.usernum1 
        self.usernum2 = args.usernum2
        self.itemnum1 = args.itemnum1
        self.itemnum2 = args.itemnum2
        self.maxlen =  args.maxlen
        self.data_type = data_type
        
        assert self.data_type in {"train", "valid", "test"}
        
        self.valid_users = self._get_valid_users(user_type)


    def _get_valid_users(self,user_type):
        valid_users = []
        
        if self.data_type == "train":
            for user in range(1, self.usernum + 1):
                

                user_in_train1 = user in self.user_train1 and len(self.user_train1[user]) > 1
                user_in_train2 = user in self.user_train2 and len(self.user_train2[user]) > 1

                
                if user_type == "target" and user_in_train1 and not user_in_train2:
                    valid_users.append(user)
                elif user_type == "overlap" and user_in_train1 and user_in_train2:
                    valid_users.append(user)
                elif user_type == "source" and not user_in_train1 and user_in_train2:
                    valid_users.append(user)
                    
        elif self.data_type == "valid" or self.data_type =="test":
            for user in range(1, self.usernum + 1):

                user_in_train1 = user in self.user_train1 and len(self.user_train1[user]) >= 2 
                user_in_train2 = user in self.user_train2 and len(self.user_train2[user]) >= 2 
                
                user_in_eva =  user in self.user_eva and len(self.user_eva[user]) >= 1
                
                if user_type == "target_overlap" and user_in_train1 and user_in_eva: 
                    valid_users.append(user)
                elif user_type == "overlap" and user_in_train1 and user_in_train2 and user_in_eva:
                    valid_users.append(user)


            select_data = False
            if select_data:

                array = np.array(valid_users)

                select_size = len(array) // 4
                selected_indices = np.random.choice(len(array), size=select_size, replace=False)

                valid_users = array[selected_indices].tolist()
            
        return valid_users
            
    def _data_sample_rec_task(self,user):
        pass
    
    def _data_sample_eval_task(self, user):
        pass
    
    def __getitem__(self, index):
        
        user = self.valid_users[index]
        
        if self.data_type == "train":
            return self._data_sample_rec_task(user)
        elif self.data_type == "valid" or self.data_type == "test":
            return self._data_sample_eval_task(user)
    
        
    def __len__(self):
        return len(self.valid_users)


class C2DSR_Dataset(baseDataset):
    def __init__(self, user_train1, user_train2,user_eva, time1, time2, args, user_type, data_type, **kwargs):
        super().__init__(user_train1, 
                         user_train2,
                         user_eva, 
                         time1, 
                         time2, 
                         args, 
                         user_type, 
                         data_type)
        self.negatvie_items = kwargs["negatvie_items"]

    def _data_sample_rec_task(self,user):

        d = np.zeros([self.maxlen], dtype=np.int32)
        xd = np.zeros([self.maxlen], dtype=np.int32)
        yd = np.zeros([self.maxlen], dtype=np.int32)
        position = np.zeros([self.maxlen], dtype=np.int32)
        x_position = np.zeros([self.maxlen], dtype=np.int32)
        y_position = np.zeros([self.maxlen], dtype=np.int32)
        

        seq1 = self.user_train1.get(user, [])[:-1]
        seq2 = self.user_train2.get(user, [])[:-1]
        time1 = self.time1.get(user, [])[:-1]
        time2 = self.time2.get(user, [])[:-1]

        mixed_seq = []
        if seq1:
            mixed_seq.extend([(item, t, 1) for item, t in zip(seq1, time1)])
        if seq2:
            mixed_seq.extend([(item, t, 2) for item, t in zip(seq2, time2)])
        mixed_seq.sort(key=lambda x: x[1])
        

        idx = self.maxlen - 1
        x_cnt = 1
        y_cnt = 1
        for item, _, domain in reversed(mixed_seq):
            if idx < 0:
                break
            
            d[idx] = item
            position[idx] = len(mixed_seq) - (self.maxlen - 1 - idx)
            
            if domain == 1:
                xd[idx] = item
                x_position[idx] = x_cnt
                x_cnt += 1
                yd[idx] = self.itemnum1 + self.itemnum2
            else:
                xd[idx] = self.itemnum1 + self.itemnum2
                yd[idx] = item
                y_position[idx] = y_cnt
                y_cnt += 1
            idx -= 1
        

        ground = np.zeros([self.maxlen], dtype=np.int32)
        share_x_ground = np.zeros([self.maxlen], dtype=np.int32)
        share_y_ground = np.zeros([self.maxlen], dtype=np.int32)
        x_ground = np.zeros([self.maxlen], dtype=np.int32)
        y_ground = np.zeros([self.maxlen], dtype=np.int32)
        
        ground_mask = np.zeros([self.maxlen], dtype=np.int32)
        share_x_ground_mask = np.zeros([self.maxlen], dtype=np.int32)
        share_y_ground_mask = np.zeros([self.maxlen], dtype=np.int32)
        x_ground_mask = np.zeros([self.maxlen], dtype=np.int32)
        y_ground_mask = np.zeros([self.maxlen], dtype=np.int32)
        

        corru_x = np.copy(xd)
        corru_y = np.copy(yd)
        

        for i in range(self.maxlen):
            if xd[i] == self.itemnum1 + self.itemnum2:
                corru_x[i] = random.randint(self.itemnum1, self.itemnum1 + self.itemnum2 - 1)
            if yd[i] == self.itemnum1 + self.itemnum2:
                corru_y[i] = random.randint(0, self.itemnum1 - 1)

        for i in range(self.maxlen-1):
            if d[i] != self.itemnum1 + self.itemnum2:
                ground[i] = d[i+1]
                ground_mask[i] = 1
                
                if d[i+1] < self.itemnum1:
                    share_x_ground[i] = d[i+1]
                    share_x_ground_mask[i] = 1
                    x_ground[i] = d[i+1]
                    x_ground_mask[i] = 1
                else:
                    share_y_ground[i] = d[i+1] - self.itemnum1
                    share_y_ground_mask[i] = 1
                    y_ground[i] = d[i+1] - self.itemnum1
                    y_ground_mask[i] = 1

        return (np.array(d, dtype=np.int64),
                np.array(xd, dtype=np.int64), 
                np.array(yd, dtype=np.int64),
                np.array(position, dtype=np.int64),
                np.array(x_position, dtype=np.int64),
                np.array(y_position, dtype=np.int64),
                np.array(ground, dtype=np.int64),
                np.array(share_x_ground, dtype=np.int64),
                np.array(share_y_ground, dtype=np.int64), 
                np.array(x_ground, dtype=np.int64),
                np.array(y_ground, dtype=np.int64),
                np.array(ground_mask, dtype=np.int64),
                np.array(share_x_ground_mask, dtype=np.int64),
                np.array(share_y_ground_mask, dtype=np.int64),
                np.array(x_ground_mask, dtype=np.int64),
                np.array(y_ground_mask, dtype=np.int64),
                np.array(corru_x, dtype=np.int64), 
                np.array(corru_y, dtype=np.int64))
    
    def _data_sample_eval_task(self, user):

        seq = np.zeros([self.maxlen], dtype=np.int32)
        xd = np.zeros([self.maxlen], dtype=np.int32)
        yd = np.zeros([self.maxlen], dtype=np.int32)
        position = np.zeros([self.maxlen], dtype=np.int32)
        x_position = np.zeros([self.maxlen], dtype=np.int32)
        y_position = np.zeros([self.maxlen], dtype=np.int32)
        

        seq1 = self.user_train1.get(user, [])
        seq2 = self.user_train2.get(user, [])
        time1 = self.time1.get(user, [])
        time2 = self.time2.get(user, [])
        

        mixed_seq = []
        if seq1:
            mixed_seq.extend([(item, t, 1) for item, t in zip(seq1, time1)])
        if seq2:
            mixed_seq.extend([(item, t, 2) for item, t in zip(seq2, time2)])
        mixed_seq.sort(key=lambda x: x[1])
        

        idx = self.maxlen - 1
        x_cnt = 1
        y_cnt = 1
        for item, _, domain in reversed(mixed_seq):
            if idx < 0:
                break
            
            seq[idx] = item
            position[idx] = len(mixed_seq) - (self.maxlen - 1 - idx)
            
            if domain == 1:
                xd[idx] = item
                x_position[idx] = x_cnt
                x_cnt += 1
                yd[idx] = self.itemnum1 + self.itemnum2  # PAD
            else:  # Domain 2
                xd[idx] = self.itemnum1 + self.itemnum2  # PAD
                yd[idx] = item
                y_position[idx] = y_cnt
                y_cnt += 1
            idx -= 1

        x_last = -1
        for id in range(len(x_position)):
            id += 1
            if x_position[-id]:
                x_last = -id
                break
            
        y_last = -1
        for id in range(len(y_position)):
            id += 1
            if y_position[-id]:
                y_last = -id
                break
        

        negative_samples = []
        for _ in range(999):
            while True:

                if self.user_eva[user][0] >= self.itemnum1:
                    sample = random.randint(0, self.itemnum2 - 1)
                    if sample != self.user_eva[user][0] - self.itemnum1:
                        negative_samples.append(sample)
                        break
                else:
                    sample = random.randint(0, self.itemnum1 - 1)
                    if sample != self.user_eva[user][0]:
                        negative_samples.append(sample)
                        break
        

        domain = 1 if self.user_eva[user][0] < self.itemnum1 else 0
        target = self.user_eva[user][0] if domain else self.user_eva[user][0] - self.itemnum1
        
        return (np.array(user, dtype=np.int64),
                np.array(seq, dtype=np.int64),
                np.array(xd, dtype=np.int64),
                np.array(yd, dtype=np.int64),
                np.array(position, dtype=np.int64),
                np.array(x_position, dtype=np.int64),
                np.array(y_position, dtype=np.int64),
                np.array(x_last, dtype=np.int64),
                np.array(y_last, dtype=np.int64),
                np.array(domain, dtype=np.int64),
                np.array(target, dtype=np.int64),
                np.array(negative_samples, dtype=np.int64))

class CDSRDataset(baseDataset):
    def __init__(self, user_train1, user_train2,user_eva, time1, time2, args, user_type, data_type, **kwargs):
        super().__init__(user_train1, 
                         user_train2,
                         user_eva, 
                         time1, 
                         time2, 
                         args, 
                         user_type, 
                         data_type)
        self.eval_only_first = args.eval_only_first

        self.negatvie_items = kwargs.get("negatvie_items", None)

    def _data_sample_rec_task(self,user):
        
        seq = np.zeros([self.maxlen], dtype=np.int64)
        seq2 = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)
        neg = np.zeros([self.maxlen], dtype=np.int64)
        t1 = np.zeros([self.maxlen], dtype=np.int64)
        t2 = np.zeros([self.maxlen], dtype=np.int64)
        neg_99 = np.zeros([99], dtype=np.int32)
        
        if user in self.user_train1:
            nxt = self.user_train1[user][-1]
            idx = self.maxlen - 1

            ts = set(self.user_train1[user])
            for i in range(99):
                neg_99[i] = random_neq(1, self.itemnum1 + 1, ts)
            for i, t in reversed(list(zip(self.user_train1[user][:-1], self.time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, self.itemnum1 + 1, ts)
                nxt = i
                if nxt > self.itemnum1:
                    print(f"errro:{ self.itemnum1}< {nxt} ")
                idx -= 1
                if idx == -1: break


        if user in self.user_train2:
            idx = self.maxlen - 1
            for i, t in reversed(list(zip(self.user_train2[user][:-1], self.time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break

        mask = np.zeros([self.maxlen], dtype=np.int32)
        if user in self.user_train1 and user in self.user_train2:
            idx2 = 0
            for idx in range(len(seq)):
                while idx2 < self.maxlen and t1[idx] >= t2[idx2]:
                    idx2 += 1
                mask[idx] = idx2


        mixed_seq = []
        mixed_time = []


        for idx in range(len(seq)):
            if seq[idx] != 0:
                mixed_seq.append(seq[idx])
                mixed_time.append(t1[idx])
            if seq2[idx] != 0:
                mixed_seq.append(seq2[idx])
                mixed_time.append(t2[idx])


        mixed = sorted(zip(mixed_seq, mixed_time), key=lambda x: x[1])
        mixed_seq, mixed_time = zip(*mixed)


        le=2*self.maxlen
        if len(mixed_seq) >le:
            mixed_seq = mixed_seq[-le:]
            mixed_time = mixed_time[-le:]


        mixed_seq = [0] * (le - len(mixed_seq)) + list(mixed_seq)
        mixed_time = [0] * (le - len(mixed_time)) + list(mixed_time)

        return  (np.array(user, dtype=np.int64 ),
                 np.array(seq, dtype=np.int64 ),
                 np.array(pos, dtype=np.int64 ),
                 np.array(neg, dtype=np.int64 ),
                 np.array(seq2, dtype=np.int64 ),
                 np.array(mask, dtype=np.int64),
                 np.array(mixed_seq, dtype=np.int64),
                 np.array(mixed_time, dtype=np.int64),
                np.array(neg_99, dtype=np.int64)
                 )
    
    def _data_sample_eval_task(self, user):
        
        seq = np.zeros([self.maxlen], dtype=np.int32)
        t1 = np.zeros([self.maxlen], dtype=np.int32)
        t2 = np.zeros([self.maxlen], dtype=np.int32)
 
        idx = self.maxlen - 1
        for item, t in reversed(list(zip(self.user_train1[user], self.time1[user]))):
            seq[idx] = item
            t1[idx] = t
            idx -= 1
            if idx == -1: break
        
     
        seq2 = np.zeros([self.maxlen], dtype=np.int32)
        mask = np.zeros([self.maxlen], dtype=np.int32) 
        

        if user in self.user_train2:
            idx = self.maxlen - 1
            for item, t in reversed(list(zip(self.user_train2[user], self.time2[user]))):
                seq2[idx] = item
                t2[idx] = t
                idx -= 1
                if idx == -1: break

            idx2 = 0
            for idx in range(len(seq)):
                while idx2 < self.maxlen and t1[idx] >= t2[idx2]:
                    idx2 += 1
                mask[idx] = idx2
            

        items = self.user_eva[user]
        
        neg_items = self.negatvie_items[user]

        mixed_seq = []
        mixed_time = []

        for idx in range(len(seq)):
            if seq[idx] != 0:
                mixed_seq.append(seq[idx])
                mixed_time.append(t1[idx])
            if seq2[idx] != 0:
                mixed_seq.append(seq2[idx])
                mixed_time.append(t2[idx])

        mixed = sorted(zip(mixed_seq, mixed_time), key=lambda x: x[1])
        mixed_seq, mixed_time = zip(*mixed)

        le = 2 * self.maxlen
        if len(mixed_seq) > le:
            mixed_seq = mixed_seq[-le:]
            mixed_time = mixed_time[-le:]


        mixed_seq = [0] * (le - len(mixed_seq)) + list(mixed_seq)
        mixed_time = [0] * (le - len(mixed_time)) + list(mixed_time)


        if self.eval_only_first:
            item_idx = items + neg_items
            if len(item_idx) != 101:
                logger.info(f"interactions: {len(items)},{items}")
                logger.info(f"number of neg_items{len(neg_items)}")
                logger.info(f"user ID: {user}, number of sample: {len(item_idx)}")
            assert len(item_idx) == 101 , "the number of items for evaluation is not 101"

        raise Exception("only predict the first item")
        return user, seq, seq2, mask, items,mixed_seq,mixed_time

class CDSRDataset_Increment(baseDataset):
    def __init__(self, user_train1, user_train2,user_eva, time1, time2, args, user_type, data_type, **kwargs):
        super().__init__(user_train1, 
                         user_train2,
                         user_eva, 
                         time1, 
                         time2, 
                         args, 
                         user_type, 
                         data_type)
        
        self.user_eva2 = kwargs["user_eva2"]
        self.negatvie_items = kwargs.get("negatvie_items", None)

        if data_type in {"valid","test"} :
            self.all_u = []
            self.all_seq = []
            self.all_seq2 = []
            self.all_mask = []
            self.all_items = []
            self.all_sampled_items = []
            self.all_mixed_seq = []
            self.all_mixed_time = []


            for user in self.valid_users:
                cur_u, cur_seq, cur_seq2, cur_mask, cur_sampled_items, cur_items,cur_mixed_seq,cur_mixed_time = self._data_sample_eval_task_increment(user)
                self.all_u.extend(cur_u)
                self.all_seq.extend(cur_seq)
                self.all_seq2.extend(cur_seq2)
                self.all_mask.extend(cur_mask)
                self.all_sampled_items.extend(cur_sampled_items)
                self.all_items.extend(cur_items)
                self.all_mixed_seq.extend(cur_mixed_seq)
                self.all_mixed_time.extend(cur_mixed_time)

    def _data_sample_rec_task(self, user):

        seq = np.zeros([self.maxlen], dtype=np.int32)
        seq2 = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        t1 = np.zeros([self.maxlen], dtype=np.int32)
        t2 = np.zeros([self.maxlen], dtype=np.int32)








        if user in self.user_train1:
            nxt = self.user_train1[user][-1]
            idx = self.maxlen - 1
            ts = set(self.user_train1[user])




            for i, t in reversed(list(zip(self.user_train1[user][:-1], self.time1[user][:-1]))):
                seq[idx] = i
                t1[idx] = t
                pos[idx] = nxt
                if nxt != 0: neg[idx] = random_neq(1, self.itemnum1 + 1, ts)
                nxt = i
                if nxt > self.itemnum1:
                    print(f"errro:{self.itemnum1}< {nxt} is not the item of this domain ")
                idx -= 1
                if idx == -1: break


        if user in self.user_train2:
            idx = self.maxlen - 1
            for i, t in reversed(list(zip(self.user_train2[user][:-1], self.time2[user][:-1]))):
                seq2[idx] = i
                t2[idx] = t
                idx -= 1
                if idx == -1:
                    break


        mask = np.zeros([self.maxlen], dtype=np.int32)
        if user in self.user_train1 and user in self.user_train2:
            idx2 = 0
            for idx in range(len(seq)):
                while idx2 < self.maxlen and t1[idx] >= t2[idx2]:
                    idx2 += 1
                mask[idx] = idx2

        mixed_seq = []
        mixed_time = []

        for idx in range(len(seq)):
            if seq[idx] != 0:
                mixed_seq.append(seq[idx])
                mixed_time.append(t1[idx])
            if seq2[idx] != 0:
                mixed_seq.append(seq2[idx])
                mixed_time.append(t2[idx])


        mixed = sorted(zip(mixed_seq, mixed_time), key=lambda x: x[1])
        mixed_seq, mixed_time = zip(*mixed)


        if len(mixed_seq) > self.maxlen:
            mixed_seq = mixed_seq[-self.maxlen:]
            mixed_time = mixed_time[-self.maxlen:]

        mixed_seq = [0] * (self.maxlen - len(mixed_seq)) + list(mixed_seq)
        mixed_time = [0] * (self.maxlen - len(mixed_time)) + list(mixed_time)

        return (np.array(user, dtype=np.int64),
                np.array(seq, dtype=np.int64),
                np.array(pos, dtype=np.int64),
                np.array(neg, dtype=np.int64),
                np.array(seq2, dtype=np.int64),
                np.array(mask, dtype=np.int64),
                np.array(mixed_seq, dtype=np.int64),
                np.array(mixed_time, dtype=np.int64))



    def _data_sample_eval_task_increment(self, user):
        eva_items_nums = len(self.user_eva[user])
        
        cur_u = []
        cur_seq = []
        cur_seq2 =[]
        cur_mask = []
        cur_items = []
        cur_sampled_items = []

        cur_mixed_seq = []
        cur_mixed_time = []

        for index in range(0, eva_items_nums):
            
            train_data1 = copy.deepcopy(self.user_train1[user]) + copy.deepcopy(self.user_eva[user][:index])
            
            seq = np.zeros([self.maxlen], dtype=np.int64)
            t1 = np.zeros([self.maxlen], dtype=np.int64)
            t2 = np.zeros([self.maxlen], dtype=np.int64)
 
            idx = self.maxlen - 1
            for item, t in reversed(list(zip(train_data1, self.time1[user]))):
                seq[idx] = item
                t1[idx] = t
                idx -= 1
                if idx == -1: break
            
            
            train_data2 = copy.deepcopy(self.user_train2[user])
            target_item = self.user_eva[user][index]
            if user in self.user_eva2:
                target_time = self.time1[user][(len(self.user_train1[user]) + index)]
                valid_source_item_cnt = 0
                for source_eva_time_index in range(len(self.user_train2[user]), len(self.time2[user])):
                    source_eva_time = self.time2[user][source_eva_time_index]
                    if source_eva_time > target_time:
                        break
                    valid_source_item_cnt += 1
                        
                train_data2 = train_data2 + copy.deepcopy(self.user_eva2[user][:valid_source_item_cnt])
                        
            seq2 = np.zeros([self.maxlen], dtype=np.int32)
            mask = np.zeros([self.maxlen], dtype=np.int32) 

            if user in self.user_train2:
                idx = self.maxlen - 1
                for item, t in reversed(list(zip(train_data2, self.time2[user]))):
                    seq2[idx] = item
                    t2[idx] = t
                    idx -= 1
                    if idx == -1: break

                idx2 = 0
                for idx in range(len(seq)):
                    while idx2 < self.maxlen and t1[idx] >= t2[idx2]:
                        idx2 += 1
                    mask[idx] = idx2
            
            neg_items = self.negatvie_items[user]
            item_idx = [target_item] + neg_items
            assert len(item_idx) == 101 , "the number of items for evaluation is not 101"


            mixed_seq = []
            mixed_time = []


            for idx in range(len(seq)):
                if seq[idx] != 0:
                    mixed_seq.append(seq[idx])
                    mixed_time.append(t1[idx])
                if seq2[idx] != 0:
                    mixed_seq.append(seq2[idx])
                    mixed_time.append(t2[idx])


            mixed = sorted(zip(mixed_seq, mixed_time), key=lambda x: x[1])
            mixed_seq, mixed_time = zip(*mixed)


            if len(mixed_seq) > self.maxlen:
                mixed_seq = mixed_seq[-self.maxlen:]
                mixed_time = mixed_time[-self.maxlen:]


            mixed_seq = [0] * (self.maxlen - len(mixed_seq)) + list(mixed_seq)
            mixed_time = [0] * (self.maxlen - len(mixed_time)) + list(mixed_time)


            cur_u.append(user)
            cur_seq.append(seq)
            cur_seq2.append(seq2) 
            cur_mask.append(mask)
            cur_items.append([target_item])
            cur_sampled_items.append(item_idx)
            cur_mixed_seq.append(mixed_seq)
            cur_mixed_time.append(mixed_time)

        return cur_u, cur_seq, cur_seq2, cur_mask, cur_sampled_items, cur_items,cur_mixed_seq,cur_mixed_time

    def __getitem__(self, index):
        
        
        if self.data_type == "train":
            user = self.valid_users[index]
            return self._data_sample_rec_task(user)

        elif self.data_type in {"valid", "test"}:
            return  self.all_u[index], self.all_seq[index],  self.all_seq2[index],self.all_mask[index],self.all_sampled_items[index],self.all_items[index],self.all_mixed_seq[index],self.all_mixed_time[index]
    
        
    def __len__(self):
        
        if self.data_type in {"valid","test"}:
            return len(self.all_items)
        
        return len(self.valid_users)
    

        
def init_dataset(dataset,args):
   
    
    if args.stage == "valid":
        user_train1 = dataset["user_train1_stage1"]
        user_train2 = dataset["user_train2_stage1"]
        user_eva = dataset["user_valid1_stage1"]
        user_eva2 = dataset["user_valid2_stage1"]
        time1 = dataset["Time1_stage1"]
        time2 = dataset["Time2_stage1"]
        
    elif args.stage == "test":
        user_train1 = dataset["user_train1_stage2"]
        user_train2 = dataset["user_train2_stage2"]
        user_eva = dataset["user_test1_stage2"]
        user_eva2 = dataset["user_test2_stage2"]
        time1 = dataset["Time1_stage2"]
        time2 = dataset["Time2_stage2"]
        
    args.usernum = dataset["usernum"]
    args.usernum1 = dataset["usernum1"]
    args.usernum2 = dataset["usernum2"]
    args.itemnum1 = dataset["itemnum1"]
    args.itemnum2 = dataset["itemnum2"]
    args.itemnum = dataset["itemnum1"]+dataset["itemnum2"]

    

    target_dataset = CDSRDataset(user_train1=user_train1,
                            user_train2=user_train2,
                            user_eva=user_eva,
                            time1 =time1,
                            time2=time2,
                            args=args,
                            user_type ="target",
                            data_type= "train"
                            )

    overlap_dataset = CDSRDataset(user_train1=user_train1,
                            user_train2=user_train2,
                            user_eva=user_eva,
                            time1 =time1,
                            time2=time2,
                            args=args,
                            user_type ="overlap",
                            data_type= "train"
                            )
    
    if len(target_dataset) > 0:
        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        target_loader = None
        
    overlap_loader = DataLoader(overlap_dataset, batch_size=args.batch_size, shuffle=True)

    eva_dataset = get_dataset(args.dataset)(user_train1=user_train1,
                            user_train2=user_train2,
                            user_eva=user_eva,
                            time1 =time1,
                            time2=time2,
                            args=args,
                            user_type = args.eva_type,
                            data_type = args.stage,
                            user_eva2=user_eva2,
                            negatvie_items=dataset["negatvie_items"]
                            )

    eva_loader = DataLoader(eva_dataset, batch_size = args.test_batch_size,shuffle=False, collate_fn=custom_collate_fn)

    return {
            "target_loader":target_loader,
            "overlap_loader":overlap_loader,
            "eva_loader":eva_loader}
    


    
