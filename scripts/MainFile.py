import sys

sys.path.append('.')

from utils.data import datapretreat, MatchDataGenerator
from utils.features import SequenceFeature, SparseFeature
from model.DSSM import DSSM
import torch
from utils.train import MatchTrainer
import numpy as np
import random
from utils.recall import topn_evaluate



def main(ProjectFolderDir, neg_ratio, min_item, seq_max_len, load, batch_size, user_params, item_params, 
            temperature, learning_rate, weight_decay, optimizer_fn, epoch, topk):
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import time
    start_time = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    logfile_name = str(time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())))
    print('Start:', start_time)
    save_dir = ProjectFolderDir  # 项目文件夹路径(推荐)
    
    model_dir = save_dir + 'temp/'    # 模型及embedding向量保存路径
    datapath = save_dir + '/data/processed/ZJULibrary2013_2019.csv'
    user_features = ['PATRON_ID', 'STUDENT_GRADE', 'PATRON_DEPT', 'PATRON_TYPE']
    # 用户侧特征       用户id          年级            学生学院       学生类型
    item_features = ['ITEM_ID', 'SUBLIBRARY', 'CALLNO1','CALLNO2', 'PUBLISH_YEAR', 'AUTHOR', 'TITLE', 'PRESS']
    # 物品侧特征       记录号      馆藏地         大类1      小类2       出版年          作者       题目    出版社
    time = 'LOAN_DATE'
    user_col = user_features[0]
    item_col = item_features[0]

    datainfo = datapretreat(save_dir, datapath, user_features, item_features, time, user_col, item_col)
    data = datainfo.data_load()
    feature_max_idx, data = datainfo.data_labelencoder(data)
    user_features = [SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name]) for feature_name in user_features]
    user_features += [SequenceFeature("hist_"+item_col, vocab_size=feature_max_idx[item_col], shared_with=item_col)]
    item_features = [SparseFeature(feature_name, vocab_size=feature_max_idx[feature_name]) for feature_name in item_features]


    item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test= \
        datainfo.get_standard_data(data, neg_ratio=neg_ratio, min_item=min_item, seq_max_len = seq_max_len, load=load)
    dg = MatchDataGenerator(item_set, user_set, x_train, y_train, x_val, y_val, x_test, y_test)
    train_dl, val_dl, test_dl, user_dl, item_dl = dg.generate_dataloader(batch_size=batch_size, num_workers=2)
    print('standard data has been generated')

    user_params={"dims": user_params}
    item_params={"dims": item_params}
    model = DSSM(user_features, item_features, user_params, item_params, temperature=temperature)
    optimizer = {"lr": learning_rate, "weight_decay": weight_decay}
    trainer = MatchTrainer(model, optimizer_fn=optimizer_fn, optimizer_params=optimizer, n_epoch=epoch, device='cuda:0', model_path=model_dir)

    eva_test = trainer.fit(train_dl, val_dl, test_dl)
    auc = eva_test['tp']+eva_test['tn']
    recall = eva_test['tp']/(eva_test['tp']+eva_test['fn'])
    precision = eva_test['tp']/(eva_test['tp']+eva_test['fp'])
    print('auc: %.4f, recall: %.4f, precision: %.4f' % (auc, recall, precision))

    user_embedding, item_embedding, topk_score = topn_evaluate(trainer, user_dl, user_set, item_dl, item_set, user_col, item_col, model, model_dir, topk)

    torch.save(user_embedding.data.cpu(), model_dir + "user_embedding.pth")
    torch.save(item_embedding.data.cpu(), model_dir + "item_embedding.pth")
    print('embedding has been saved to '+model_dir)

    import time
    finish_time = str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    print('Finish:', finish_time)
    log = open(save_dir+'/log/'+logfile_name+'.txt', 'a')
    log.write(
        'Start: ' + start_time + '\n'+
        'datapath: '+ datapath + '\n' +
        'neg_ratio: ' + str(neg_ratio) + '\n'+
        'min_item: ' + str(min_item) + '\n'+
        'seq_max_len: ' + str(seq_max_len) +'\n'+
        'load: ' + str(load) + '\n'+
        'batch_size: ' + str(batch_size) + '\n'+
        'user_params: ' + str(user_params) + '\n'+
        'item_params: ' + str(item_params) + '\n'+
        'temperature: ' + str(temperature) + '\n'+
        'learning_rate: ' + str(learning_rate) + '\n'+
        'weight_decay: ' + str(weight_decay) + '\n'+
        'optimizer_fn: ' + str(optimizer_fn) + '\n'+
        'epoch: ' + str(epoch) + '\n'+
        'evaluation'+str(eva_test) + '\n'
        'auc: ' + str(auc) + '\n'+
        'recall: ' + str(recall) + '\n'+
        'topk_score: ' + str(topk_score) + '\n'
        'precision: ' + str(precision) + '\n'+
        'model: ' + str(model) + '\n'+
        'Finish:' + finish_time + '\n\n'
    )
    log.close()
    return 0

if __name__ == '__main__':
    main(
        ProjectFolderDir = '',
        neg_ratio= 3 ,
        min_item= 5 ,
        seq_max_len= 10 ,
        load= False , 
        batch_size= 2048 ,
        user_params= [512, 512, 256, 128, 64] ,
        item_params= [1024, 512, 256, 128, 64] ,
        temperature= 0.02 ,
        learning_rate= 0.01 ,
        weight_decay= 1e-4 ,
        optimizer_fn= torch.optim.Adam ,
        epoch= 10 ,
        topk= 100
    )
    
