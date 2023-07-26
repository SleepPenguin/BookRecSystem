import pandas as pd
import re
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import  random
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def get_Bletter(str0):   # 取出大写字母
    b = re.sub(u"([^\u0041-\u007a])", "", str0)
    return b

def df_to_dict(data):
    data_dict = data.to_dict('list')
    for key in data.keys():
        data_dict[key] = np.array(data_dict[key])
    return data_dict

def pad_sequences(sequences, maxlen=20, dtype='int32', value=0.):
    arr = np.full((len(sequences), maxlen), value, dtype=dtype)  # 这里相当于增加了第0类item
    for idx, x in enumerate(sequences):
        if len(x) == 0:
            continue  # empty list
        else:
            trunc = x[-maxlen:]
        trunc = np.asarray(trunc, dtype=dtype)
        arr[idx, -len(trunc):] = trunc
    return arr

def gen_model_input(df, user_profile, user_col, item_profile, item_col, seq_max_len):
    df = pd.merge(df, user_profile, on=user_col, how='left')
    df = pd.merge(df, item_profile, on=item_col, how='left')
    df['hist_ITEM_ID'] = pad_sequences(df['hist_ITEM_ID'], maxlen=seq_max_len, value=0).tolist()
    if 'pos_list' in list(df):
        df['pos_list'] = pad_sequences(df['pos_list'], maxlen=100, value=0).tolist()
    input_dict = df_to_dict(df)
    return input_dict

class datapretreat(object):

    def __init__(self, save_dir, datapath, user_features,item_features, time, user_col, item_col):
        self.savepath = save_dir + '/data/processed/'
        self.datapath = datapath
        self.user_features = user_features
        self.item_features = item_features
        self.features = user_features + item_features
        self.time = time
        self.user_col = user_col
        self.item_col = item_col

    def data_load(self):
        data = pd.read_csv(self.datapath, encoding='utf8', dtype=str)
        data[self.time]=data[self.time].astype(int)
        return data[self.features+[self.time]]

    def data_labelencoder(self, data):
        feature_max_idx = {}    # 记录每个特征共有多少种
        for feature in self.features:
            lbe = LabelEncoder()
            data[feature] = lbe.fit_transform(data[feature]) + 1  # 从1开始
            feature_max_idx[feature] = data[feature].max() + 1 # 防止padding后造成分类标签越界
            if feature == self.user_col:
                user_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
            if feature == self.item_col:
                item_map = {encode_id + 1: raw_id for encode_id, raw_id in enumerate(lbe.classes_)}
        np.save(os.path.join(self.savepath, "raw_id_maps.npy") ,np.array((user_map, item_map), dtype=object))
        return feature_max_idx, data

    def get_standard_data(self, data, neg_ratio = 2, min_item = 2, seq_max_len = 20, load = False):
        user_profile = data[self.user_features].drop_duplicates(self.user_col)
        item_profile = data[self.item_features].drop_duplicates(self.item_col)
        # 阅读序列特征抽取
        if load:
            item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test = tqdm.tqdm(np.load(self.savepath+'data_process.npy', allow_pickle=True), desc='loading data:')
        else:
            user_col = self.user_col
            item_col = self.item_col
            data.sort_values(self.time, inplace=True)
            items = item_profile[self.item_col].tolist()
            data_set = []
            user_set = []
            n_cold_user = 0  
            

            for uid, hist in tqdm.tqdm(data.groupby(self.user_col), desc='generate train set, validation set and test set:'):
                pos_list = hist[self.item_col].tolist()
                if len(pos_list) < min_item:  # drop this user when his pos items < min_item
                    n_cold_user += 1
                    continue
                for i in range(0, len(pos_list)):
                    hist_item = pos_list[:i]
                    sample = [uid, pos_list[i], hist_item, len(hist_item)]
                    data_set.append(sample + [1])
                    for _ in range(neg_ratio):
                        sample[1] = random.choice(items)
                        data_set.append(sample + [0])
                    if i == len(pos_list)-1:
                        hist_item = pos_list[:i]
                        sample = [uid, pos_list[i], hist_item, len(hist_item), pos_list]
                        user_set.append(sample + [1])                       
            train_val_set, test_set = train_test_split(data_set, test_size=0.1, random_state=2023)
            train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=2023)
            print("n_train: %d, n_val: %d, n_test: %d" % (len(train_set), len(val_set), len(test_set)))
            print("%d cold start user droped " % (n_cold_user))

            df_train = pd.DataFrame(train_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col, 'label'])
            df_val = pd.DataFrame(val_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col, 'label'])
            df_test = pd.DataFrame(test_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col, 'label'])
            df_user = pd.DataFrame(user_set, columns=[user_col, item_col, "hist_" + item_col, "histlen_" + item_col, 'pos_list', 'label'])
            
            item_set = df_to_dict(item_profile)
            user_set = gen_model_input(df_user, user_profile, user_col, item_profile, item_col, seq_max_len)
            x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len)
            y_train = x_train.pop("label")
            x_val = gen_model_input(df_val, user_profile, user_col, item_profile, item_col, seq_max_len)
            y_val = x_val.pop("label")
            x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len)
            y_test = x_test.pop("label")

            np.save(os.path.join(self.savepath, "data_process.npy"), np.array((item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test), dtype=object))
            np.save(os.path.join(self.savepath, "item_user.npy"), np.array((item_set, user_set), dtype=object))
            print('train set, validation set and test set have saved in data/processed/data_process.npy')

        return item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test


class PredictDataset(Dataset):

    def __init__(self, x):
        super().__init__()
        self.x = x

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}

    def __len__(self):
        return len(self.x[list(self.x.keys())[0]])



class TorchDataset(Dataset):

    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.x.items()}, self.y[index]

    def __len__(self):
        return len(self.y)


class MatchDataGenerator(object):

    def __init__(self, item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test):
        super().__init__()
        self.train_dataset = TorchDataset(x_train, y_train)
        self.val_dataset = TorchDataset(x_val, y_val)
        self.test_dataset = TorchDataset(x_test, y_test)
        self.user_dataset = PredictDataset(user_set)  
        self.item_dataset = PredictDataset(item_set)   

    def generate_dataloader(self, batch_size, num_workers=2):
        train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        # shuffle = False to keep the same order for matching embedding   
        user_dataloader = DataLoader(self.user_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        item_dataloader = DataLoader(self.item_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader, user_dataloader, item_dataloader
