import csv
from os import read
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import scale,normalize
from sklearn.decomposition import PCA
from statistics import product_dict

def load_data(dataset,method,epoch,protein_num):# name = "AT"
    print(dataset)
    print(method)
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset + '/coding/' + method + '.csv') as f:
        reader = csv.reader(f)
        head_row = next(reader)#去掉第一行
        dict1 = product_dict(dataset)[0]
        feature_dim =len(head_row[0].split('\t')) -1
        feature = np.zeros((protein_num, feature_dim))
        for line in reader:
            line = line[0].split('\t')
            for i in range(len(line)-1): #编码特征写入特征矩阵，-1是因为第一个数据是蛋白质序列所以数据长度减1
                if line[0] in dict1:
                    feature[dict1[line[0]]-1][i] =float(line[i+1])#前面的-1是因为蛋白质质编号是从1开始的，后面的+1是因为第一个数据是序列所以第一个不要
    #读入SNLF模型产生的数据
    with open ('C:/Users/Administrator/Desktop/PASNVGA/dataset/SNLF_feature/'+ str(epoch) + 'pos_' + dataset + '_featurematrix.txt') as f:
        weidu = f.readline()
        sec_feature_dim= len(weidu.split('::'))-1    
    with open ('C:/Users/Administrator/Desktop/PASNVGA/dataset/SNLF_feature/' + str(epoch) + 'pos_' + dataset + '_featurematrix.txt') as f:
        sec_feature = f.readlines()
    r = 0
    feature_1 = np.zeros((len(sec_feature),sec_feature_dim))
    for line in sec_feature:
        x = line.split('::')
        for i in range(len(x)-1):
            feature_1[r][i] = float(x[i])
        r = r + 1
    features = np.hstack((feature_1,feature))
    #加入了PCA
    pca = PCA(n_components=64)
    pca.fit(features)
    features=pca.fit_transform(features)
    features = sp.csr_matrix(features)

    # QSOrder codings for proteins
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/'+ dataset + '/pos.txt') as f:
        adj = f.readlines()
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset + '/' + str(epoch) + 'pos_' + dataset + '_train.txt') as f:# 前面的1表示第一份
        postrain = f.readlines()
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset + '/' + str(epoch) + 'neg_' + dataset + '_train.txt') as f:
        negtrain = f.readlines()
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset + '/' + str(epoch) + 'pos_' + dataset + '_test.txt') as f:
        postest = f.readlines()
    with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset + '/' + str(epoch) + 'neg_' + dataset + '_test.txt') as f:
        negtest = f.readlines()
    num_node = feature.shape[0]
    adjs = np.zeros((num_node,num_node)) #形状会变
    for line in adj:
        x,y,z = line.split("::")
        adjs[int(x)-1][int(y)-1] = 1   
    adjs = adjs + adjs.T
    adj = sp.csr_matrix(adjs)
    adj_train = np.zeros((num_node,num_node))

    for line in postrain:
        x,y,z = line.split("::")
        adj_train[int(x)-1][int(y)-1] = 1
        adj_train[int(y)-1][int(x)-1] = 1
    adj_train = sp.csr_matrix(adj_train)

    postrain_rows = []
    postrain_cols = []   
    for line in postrain:
        x,y,z = line.split("::")
        postrain_rows.append(int(x)-1)
        postrain_cols.append(int(y)-1)
    X = np.array(postrain_rows)
    Y = np.array(postrain_cols)
    train_edges = np.vstack((X,Y)).transpose()
    negtrain_rows = []
    negtrain_cols = []

    for line in negtrain:
        x,y,z = line.split("::")
        negtrain_rows.append(int(x)-1)
        negtrain_cols.append(int(y)-1)
    X = np.array(negtrain_rows)
    Y = np.array(negtrain_cols)
    train_edges_false = np.vstack((X,Y)).transpose()

    postest_rows = []
    postest_cols = []
    for line in postest:
        x,y,z = line.split("::")
        postest_rows.append(int(x)-1)
        postest_cols.append(int(y)-1)
    X = np.array(postest_rows)
    Y = np.array(postest_cols)
    test_edges = np.vstack((X,Y)).transpose()

    negtest_rows = [] 
    negtest_cols = []
    for line in negtest:
        x,y,z = line.split("::")
        negtest_rows.append(int(x)-1)
        negtest_cols.append(int(y)-1)
    X = np.array(negtest_rows)
    Y = np.array(negtest_cols)
    test_edges_false= np.vstack((X,Y)).transpose()
    
    
    return adj, adj_train,  features, train_edges, train_edges_false,  test_edges, test_edges_false


if __name__ == "__main__":
    adj, adj_train,   features, train_edges, train_edges_false,  test_edges, test_edges_false = load_data("AT", 'QSOrder',1,756)

   