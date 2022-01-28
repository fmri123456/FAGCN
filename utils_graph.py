import numpy as np
import torch
import scipy.sparse as sp
from scipy import sparse
import load_data

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = np.array(list(map(classes_dict.get,labes)), dtype=np.int32)
    #print(labes_onehot.shape)
    return labes_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_data1():
    print("load dataset ..... ")
    # _, _, feature, label = utils.load()
    train_feat, train_label,train_wei,train_sim,test_feat,test_label,test_wei,test_sim = load_data.load()
    # train_feat, train_label, feature, label,adj = load_mao_data.load()
    train_feat = torch.FloatTensor(train_feat)
    train_label = onehot_encode(train_label)
    train_label = torch.LongTensor(np.where(train_label)[1])
    train_wei = torch.FloatTensor(train_wei)
    train_sim = torch.FloatTensor(train_sim)

    test_feat = torch.FloatTensor(test_feat)
    test_label = onehot_encode(test_label)
    test_label = torch.LongTensor(np.where(test_label)[1])
    test_wei = torch.FloatTensor(test_wei)
    test_sim = torch.FloatTensor(test_sim)


    # test_feat = torch.from_numpy(feature)
    # test_labels = onehot_encode(label)
    # test_label = torch.LongTensor(np.where(test_labels)[1])
    #
    # # adj = sp.coo_matrix(adj)
    # features = sp.csr_matrix(train_feat, dtype=np.float32)
    # labels = onehot_encode(label)
    #adj =np.load("adj.npy")
    # adj1 = adj[1]
    # adj1 = adj1.reshape(90,90)
    # adj1 = sp.coo_matrix(adj1)
    # adj3 = sparse_mx_to_torch_sparse_tensor(adj1)
    # adj = sparse.load_npz(filename+'adj_sp.npz')
    #print(adj.shape)
    # adj = normalize(adj + sp.eye(adj.shape[0]))
    # idx_train = range(10)
    # idx_val = range(10)
    # idx_test = range(10)

    # features = torch.FloatTensor(np.array(features.todense()))
    # features = torch.reshape(features,(-1,90,60))
    # labels = torch.LongTensor(np.where(labels)[1])
    # adj = torch. torch.FloatTensor(adj)
    # # adj = sparse_mx_to_torch_sparse_tensor(adj)
    # adj = torch.reshape(adj, (-1, 90, 90))
    #
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    # print(idx_test)
    return  train_feat, train_label,train_wei,train_sim, test_feat,test_label,test_wei,test_sim

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def stastic_indicators(output,labels):
    TP = ((output.max(1)[1] == 1) & (labels == 1)).sum()
    TN = ((output.max(1)[1] == 0) & (labels == 0)).sum()
    FN = ((output.max(1)[1] == 0) & (labels == 1)).sum()
    FP = ((output.max(1)[1] == 1) & (labels == 0)).sum()
    return TP,TN,FN,FP
if __name__ == '__main__':
    train_feat, train_label,train_wei,train_sim, test_feat,test_label,test_wei,test_sim = load_data()
    #print(idx_test)