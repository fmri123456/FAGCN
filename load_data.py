import scipy.io as sio
import numpy as np
# dataFile1 = '0_载入数据.mat'
# dataFile2 = '2_阈值筛选.mat'
dataFile3 = 'pearson_NC.AD_双.mat'
dataFile4 = 'feat_1.mat'
# data1 = sio.loadmat(dataFile1)
# data2 = sio.loadmat(dataFile2)
data3 = sio.loadmat(dataFile3)
data4 = sio.loadmat(dataFile4)
# feat_NC = data1['NC_35']
# feat_AD = data1['AD_35']
# weigh_NC = data2['NC']
# weigh_AD = data2['AD']
# adj_AD = data2['adj_AD']
# adj_NC = data2['adj_NC']
# adj = np.concatenate((adj_NC,adj_AD))
# print(weigh_AD.shape)
# print(adj_AD.shape)
#----------------------------------
Feat = data3['feat']
Wei = data3['w']
Sim = data3['Sim']

# Wei = data4['w0']
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma



label = np.concatenate((np.ones(233),np.zeros(237)))
shuffle_idx = np.array(range(0, 470))
np.random.shuffle(shuffle_idx)
feat = Feat[shuffle_idx]
labels = label[shuffle_idx]
wei = Wei[shuffle_idx]
sim = Sim[shuffle_idx]
train_id = range(0,420)
test_id = range(420,470)

# label = np.concatenate((np.ones(233),np.zeros(203)))
# shuffle_idx = np.array(range(0, 436))
# np.random.shuffle(shuffle_idx)
# feat = Feat[shuffle_idx]
# labels = label[shuffle_idx]
# wei = Wei[shuffle_idx]
# sim = Sim[shuffle_idx]

# train_id = range(0,392)
# test_id = range(392,436)

# def load():
#     return data[train_idx], label[train_idx], data[test_idx], label[test_idx],adj[train_idx]
def load():
    return feat[train_id],labels[train_id],wei[train_id],sim[train_id],feat[test_id],labels[test_id],wei[test_id],sim[test_id]