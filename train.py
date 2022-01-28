import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 不用gpu，cuda有点问题
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import GCN
import utils_graph
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import roc_curve, auc



train_feat, train_label,train_wei,train_sim, test_feat,test_label,test_wei,test_sim= utils_graph.load_data1()
# print(adj.shape)
# print(features.shape)



import torch.utils.data as Data
batch_size = 16
dataset = Data.TensorDataset(train_wei,train_sim, train_feat, train_label)

train_loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


import torch.nn as nn
import torch.nn.functional as F
model = GCN(70,80,2)
#print(model)

LR  = 0.00001
EPOCH = 101
max_acc = 0
loss_list = []
acc_list = []


out_data = torch.zeros(420,2)



optimizer = optim.Adam(model.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (b_w,b_sim,b_feat, b_lab) in enumerate(train_loader):
        output = model.forward(b_w,b_feat,b_sim)
        loss = loss_func(output, b_lab)
        acc_val = utils_graph.accuracy(output, b_lab)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()

        if epoch == EPOCH-1:
            if output.shape[0]==batch_size:
                out_data[step*16:(step+1)*output.shape[0],:] = output



    if epoch % 10 == 0:
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.4f}'.format(loss.item()),
                'acc_val: {:.4f}'.format(acc_val.item()))

        model.eval()
        output1 = model(test_wei, test_feat,test_sim)
        loss_val1 = nn.CrossEntropyLoss()(output1, test_label)
        acc_val1 = utils_graph.accuracy(output1, test_label)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("Test set results:",
              "loss= {:.4f}".format(loss_val1.item()),
              "accuracy= {:.4f}".format(acc_val1.item()))
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        loss_list.append(float(loss_val1.item()))
        acc_list.append(float(acc_val1.item()))
    if max_acc < acc_val1:
        max_acc = acc_val1
        TP , TN , FN , FP = utils_graph.stastic_indicators(output1, test_label)
        ACC = (TP + TN) / (TP + TN + FP + FN)
        SEN = TP / (TP + FN)
        SPE = TN /(FP + TN)
        BAC = (SEN + SPE) / 2
        output2 = output1

# print(test_label)
#特征提取
fc1_w = model.state_dict()['fc1.weight']
fc2_w = model.state_dict()['fc2.weight']
fc3_w = model.state_dict()['fc3.weight']
fc4 = out_data[0:20]
best_imp,best_idx = fre_statis(fc1_w , fc2_w , fc3_w , fc4)
for num in range(best_idx.shape[0]):
    if fc4[num,0]<fc4[num,1]:
        best_index = best_idx[num]
        best_important = best_imp[num]
        break


# #画ROC曲线
# y_test = test_label.numpy()
#
# y_score = F.softmax(output1,1)
# y_score = y_score.detach().numpy()
# y_scores = y_score[0:47,1]
# # for i in range(y_score.shape[0]):
# #     if y_test[i] == 0:
# #         y_scores[i] = y_score[i, 0]
# #     else:
# #         y_scores[i] = y_score[i ,1]
# fpr,tpr,thr = roc_curve(y_test,y_scores)
#
#
# # tpr = np.load('TPR.npy')
# # fpr = np.load('FPR.npy')
#
# roc_auc = auc(fpr,tpr)
# lw = 2
# plt.figure(figsize=(10 , 10))
# plt.plot(fpr , tpr , color='darkorange' ,
#          lw=lw , label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0 , 1] , [0 , 1] , color='navy' , lw=lw , linestyle='--')
# plt.xlim([0, 1.0])
# plt.ylim([0 , 1.01])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
#
# plt.show()
#
# #保存值
# np.save('acc_list.npy',acc_list)
# np.save('TP.npy',TP)
# np.save('FP.npy',FP)
# np.save('TN.npy',TN)
# np.save('FN.npy',FN)
# np.save('TPR.npy',tpr)
# np.save('FPR.npy',fpr)
# np.save('best_imp.npy',best_imp)
# np.save('best_idx.npy',best_idx)
#
