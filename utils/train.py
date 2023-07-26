import os
import torch
import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np


class MatchTrainer(object):
    def __init__(self, model, optimizer_fn, optimizer_params=None, n_epoch=10, device="cuda:0", model_path="./"):
        self.model = model  # for uniform weights save method in one gpu or multi gp
        self.device = torch.device(device)
        self.model.to(self.device)
        self.criterion = torch.nn.BCELoss()  # default loss binary cross_entropy
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.n_epoch = n_epoch
        self.model_path = model_path

    def train_one_epoch(self, train_data_loader, val_data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(train_data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device)
            y = y.float()  # torch._C._nn.binary_cross_entropy expected Float
            y_pred = self.model(x_dict)
            loss = self.criterion(y_pred, y)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)  # 输出平均值
                total_loss = 0
        train_data_auc = self.evaluate(self.model, train_data_loader, desc = 'calculate auc on train data') # 初步判断模型拟合效果
        val_data_auc = self.evaluate(self.model, val_data_loader, desc = 'calculate auc on validation data')
        print('auc on train data:', train_data_auc)
        print('auc on validation data:' , val_data_auc)

    def fit(self, train_dataloader, val_dataloader, test_dataloader):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader, val_dataloader)
        eva_test_data = self.evaluate(self.model, test_dataloader, desc = 'calculate confusion matrix on test data')
        print('\nconfusion_matrix on test data:', eva_test_data)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))
        return eva_test_data


    def inference_embedding(self, model, mode, data_loader, model_path):
        # inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        model.mode = mode
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))
        model = model.to(self.device)
        model.eval()
        predicts = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="%s inference" %(mode), smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_pred = model(x_dict)
                predicts.append(y_pred.data)
        return torch.cat(predicts, dim=0)


    def evaluate(self, model, data_loader, desc):
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc=desc, smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
            if 'test' in desc: 
                predicts = np.around(predicts,0).astype(int).tolist()
                tn, fp, fn, tp = confusion_matrix(targets, predicts).ravel()
                result0 = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
                result = {key:value/len(targets) for key,value in result0.items()}
            else:
                result = self.evaluate_fn(targets, predicts)
        return result