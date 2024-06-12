import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader


def load_data(root_dir, tm_fn, rm_fn, n_train, n_test, known_rate, b_size, scale=10**9, model_name='autotomo', dataset_name='abilene'):
    df_tm = pd.read_csv(os.path.join(root_dir, tm_fn), header=None)
    if dataset_name != 'cernet':
        df_tm.drop(df_tm.columns[-1], axis=1, inplace=True)
    traffic = torch.from_numpy(df_tm.values / scale).float()
    _, feature_size = traffic.shape

    df_rm = pd.read_csv(os.path.join(root_dir, rm_fn), header=None)
    df_rm.drop(df_rm.columns[-1], axis=1, inplace=True)
    rm = torch.from_numpy(df_rm.values).float()

    link_loads = traffic @ rm

    if model_name == 'autotomo-os':
        known_train_id = flow_selection(link_loads, rm, known_train_rate=known_rate)
    else:
        known_train_id = add_unknown(feature_size, known_train_rate=known_rate)
    
    traffic_train = get_proccessed_data(traffic, known_train_id)
    traffic_train = traffic_train.float()

    if dataset_name == 'geant':
        train_size = int(n_train * 96)
        test_size = int(n_test * 96)
    else:
        train_size = int(n_train * 288)
        test_size = int(n_test * 288)

    if train_size + test_size > link_loads.shape[0]:
        up_constraint = link_loads.shape[0]
    else:
        up_constraint = train_size + test_size

    train_flow = traffic_train[0:train_size]
    test_flow = traffic[train_size:up_constraint]

    train_link = link_loads[0:train_size]
    test_link = link_loads[train_size:up_constraint]

    train_loader = DataLoader(dataset=TMEDataset(train_flow, train_link), batch_size=b_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=TMEDataset(test_flow, test_link), batch_size=b_size, shuffle=False, num_workers=0)
    
    real_train_loader = DataLoader(dataset=TMEDataset(traffic[0:train_size], train_link), batch_size=b_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, rm, known_train_id, real_train_loader


def add_unknown(feature_size, known_train_rate):
    known_train_num = int(np.ceil(feature_size * known_train_rate))
    id_rdm = torch.randperm(feature_size)
    known_train_id = id_rdm[0:known_train_num]
    id_rdm = torch.randperm(known_train_num)
    known_train_id = known_train_id[id_rdm]
    return known_train_id


def flow_selection(link_loads_tensor, rm_tensor, known_train_rate):
    known_train_num = int(np.ceil(rm_tensor.shape[0] * known_train_rate))
    flow_n = rm_tensor.shape[0]
    link_mean = torch.mean(link_loads_tensor, 0)
    f_min = torch.zeros(1, flow_n)
    known_train_id = torch.zeros(known_train_num)
    r_tensor_sp = torch.clone(rm_tensor)

    for i in range(flow_n):
        f_min_i = torch.min(link_mean[rm_tensor[i, :] > 0])
        f_min[0, i] = f_min_i

    _, f_sort_id = torch.sort(f_min, dim=1, descending=True)
    k = 0
    i = 0
    f_id_all = f_sort_id[0, :].numpy()

    while i < known_train_num:
        if k == len(f_id_all):
            known_train_id[i:flow_n] = torch.from_numpy(f_id_all[0:known_train_num - i])
            break
        else:
            f_id = f_id_all[k]
            f_m = torch.zeros(flow_n, 1)
            f_m[f_id, 0] = 1
            r = torch.linalg.matrix_rank(r_tensor_sp)
            r1 = torch.linalg.matrix_rank(torch.cat((r_tensor_sp, f_m), 1))

            if r1 > r:
                known_train_id[i] = f_id
                i = i + 1
                f_id_all = np.delete(f_id_all, k)
                r_tensor_sp = torch.cat((r_tensor_sp, f_m), 1)
            else:
                k = k + 1

    return known_train_id.long()


def get_proccessed_data(data, known_train_id):
    feature_size = data.shape[1]
    known_data_mean = torch.mean(data[:, known_train_id], dim=1).view(-1, 1)
    data_train_tensor = known_data_mean.repeat(1, feature_size)
    data_train_tensor[:, known_train_id] = data[:, known_train_id]

    return data_train_tensor


class TMEDataset(torch.utils.data.Dataset):
    def __init__(self, traffic_matrix, link_load):
        super(TMEDataset, self).__init__()
        self.data = link_load
        self.label = traffic_matrix
        self.sample_num = traffic_matrix.shape[0]
        self.feat_dim = traffic_matrix.shape[-1]
        self.link_dim = link_load.shape[-1]

    def __getitem__(self, ind):
        return self.data[ind, :], self.label[ind, :]

    def __len__(self):
        return self.sample_num
