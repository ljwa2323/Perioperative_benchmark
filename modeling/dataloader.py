import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class INSPIRE(object):

    def __init__(self, root_path, ds, id_col, param_path=None):

        super().__init__()
        self.root_path = root_path
        self.all_id = ds.loc[:, id_col].tolist()
        print("Number of samples:", len(self.all_id))
        self.param_path = param_path
        if param_path:
            self.stat_lab, self.stat_vit, self.stat_x_s, self.stat_ward_vit = self.get_stat_info(param_path)

    def get_stat_info(self, path):
        stat_lab = torch.tensor(pd.read_csv(os.path.join(path, "z_param_lab.csv"), header=0).iloc[:,1:].values).float()
        stat_vit = torch.tensor(pd.read_csv(os.path.join(path, "z_param_vit.csv"), header=0).iloc[:,1:].values).float()
        stat_x_s = torch.tensor(pd.read_csv(os.path.join(path, "z_param_static.csv"), header=0).iloc[:,1:].values).float()
        stat_ward_vit = torch.tensor(pd.read_csv(os.path.join(path, "z_param_ward_vit.csv"), header=0).iloc[:,1:].values).float()
        return stat_lab, stat_vit, stat_x_s, stat_ward_vit

    def normalize(self, x, m, s):
        return (x - m) / s

    def unnormalize(self, x, m, s):
        return (x * s) + m

    def len(self):
        return len(self.all_id)

    def get_1data(self, ind, normalize=False):

        folder_path = os.path.join(self.root_path, str(self.all_id[ind]))

        ds_x_s = pd.read_csv(os.path.join(folder_path, "x_s.csv"), header=0)
        ds_t_list = pd.read_csv(os.path.join(folder_path, "t_list.csv"), header=0)

        ds_lab = pd.read_csv(os.path.join(folder_path, "lab.csv"), header=0)
        ds_mask_lab = pd.read_csv(os.path.join(folder_path, "mask_lab.csv"), header=0)

        ds_vit = pd.read_csv(os.path.join(folder_path, "vit.csv"), header=0)
        ds_mask_vit = pd.read_csv(os.path.join(folder_path, "mask_vit.csv"), header=0)

        ds_ward_vit = pd.read_csv(os.path.join(folder_path, "ward_vit.csv"), header=0)
        ds_mask_ward_vit = pd.read_csv(os.path.join(folder_path, "mask_ward_vit.csv"), header=0)

        ds_y_mat = pd.read_csv(os.path.join(folder_path, "y_mat.csv"), header=0)
        ds_y_mask = pd.read_csv(os.path.join(folder_path, "y_mask.csv"), header=0)
        ds_y_static = pd.read_csv(os.path.join(folder_path, "y_static.csv"), header=0)
        ds_y_mask1 = pd.read_csv(os.path.join(folder_path, "y_mask1.csv"), header=0)

        lab = torch.tensor(ds_lab.iloc[:,1:].values).float()
        mask_lab = torch.tensor(ds_mask_lab.iloc[:,1:].values).float()

        vit = torch.tensor(ds_vit.iloc[:,1:].values).float()
        mask_vit = torch.tensor(ds_mask_vit.iloc[:,1:].values).float()

        ward_vit = torch.tensor(ds_ward_vit.iloc[:,1:].values).float()
        mask_ward_vit = torch.tensor(ds_mask_ward_vit.iloc[:,1:].values).float()

        t_vit = torch.tensor(ds_vit.iloc[:,0:1].values).float()
        t_list = torch.tensor(ds_t_list.iloc[:,0:1].values).float()
        x_s =  torch.tensor(ds_x_s.values).float()
        y_mat =  torch.tensor(ds_y_mat.values).float()
        y_mask =  torch.tensor(ds_y_mask.values).float()
        y_static =  torch.tensor(ds_y_static.values).float()
        y_mask1 =  torch.tensor(ds_y_mask1.values).float()
        if normalize and self.param_path is not None:
            x_s = self.normalize(x_s, self.stat_x_s[:, 0], self.stat_x_s[:, 1])
            lab = self.normalize(lab, self.stat_lab[:, 0], self.stat_lab[:, 1])
            vit = self.normalize(vit, self.stat_vit[:, 0], self.stat_vit[:, 1])
            ward_vit = self.normalize(ward_vit, self.stat_ward_vit[:, 0], self.stat_ward_vit[:, 1])

        return lab, mask_lab, vit, mask_vit, ward_vit, mask_ward_vit, \
                t_vit, t_list, x_s, \
                y_mat, y_mask, y_static, y_mask1

    def get_batch_data(self, inds, normalize=False):

        batches = []
        ids1 = []
        for i in range(len(inds)):
            data = self.get_1data(inds[i], normalize)
            if data is None:
                continue
            else:
                batches.append(data)
                ids1.append(inds[i])
        return batches, ids1

    def iterate_batch(self, size, shuffle=True, normalize=False):

        if size > self.len():
            size = self.len()  # 如果batch size大于总样本数，设置为总样本数

        if shuffle:
            all_ids = np.random.choice(self.len(), size=self.len(), replace=False)
        else:
            all_ids = list(range(self.len()))

        if self.len() % size == 0:
            n = self.len() // size
        else:
            n = self.len() // size + 1

        for i in range(n):
            if i == n - 1:  # 修改条件，确保最后一个batch正确处理
                yield self.get_batch_data(all_ids[(size * i):], normalize)
            else:
                yield self.get_batch_data(all_ids[(size * i): (size * (i + 1))], normalize)
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect one generated INSPIRE sample")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    operation_table = pd.read_csv(os.path.join(args.data_dir, "operation_.csv"), header=0)
    dataset = INSPIRE(
        os.path.join(args.data_dir, "all_op_id"),
        operation_table[operation_table["dataset"] == 1],
        id_col="op_id",
        param_path=os.path.join(args.data_dir, "param_folder"),
    )
    dataset.get_1data(args.index, normalize=True)

