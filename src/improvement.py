from ast import Break
import csv
from math import degrees
from os import read
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from statistics import product_dict


def Ehance_Matrix (dataset,method,protein_num):# name = "AT"
    with open('/home/wangliwei/S-VGAE/iFeature_dataset/' + dataset + '/' + method + '.csv') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        dict1 = product_dict(dataset)[0]
        feature_dim =len(head_row[0].split('\t')) -1
        feature = np.zeros((protein_num, feature_dim)) 
        for line in reader:
            line = line[0].split('\t')
            for i in range(len(line)-1):
                if line[0] in dict1:
                    feature[dict1[line[0]]-1][i] =float(line[i+1])
                
    with open('/home/wangliwei/java/smalldata/' + dataset + '/pos.txt') as f:
        adj = f.readlines()
    num_node = feature.shape[0]
    adjs = np.zeros((num_node,num_node)) #形状会变
    for line in adj:
        x,y,z = line.split("::")
        adjs[int(x)-1][int(y)-1] = 1   
    adjs = adjs + adjs.T
    rowsum = np.array(adjs.sum(1))
    for i in range(len(rowsum)):
        if rowsum[i] != 0:
            rowsum[i] = 1/rowsum[i]
    degree_M = np.diag(rowsum)
    Strcture_Trasition = np.dot(degree_M,adjs)
    L3_adj = sp.coo_matrix(np.dot(np.dot(adjs,adjs),adjs))
    L3_coords = np.vstack((L3_adj.row, L3_adj.col)).transpose()
    adjs= sp.coo_matrix(adjs)
    adj_coords = np.vstack((adjs.row, adjs.col)).transpose()
    Attribute_Trasition = np.zeros((num_node,num_node))
    
    for pair in L3_coords:
        Attribute_Trasition[pair[0],pair[1]] = cosine_similarity(feature[pair[0],:],feature[pair[1],:])
    alpha = 0.5 
    delta = 2.0
    t = 20
    Probability = Proximity_Calculation(Attribute_Trasition,Strcture_Trasition,alpha,adj_coords)
    First_Probability = Probability
    if t != 1:
        for i in range(t-1):
            Probability = np.power(delta,-(i + 1))*(alpha * np.dot(Probability, Strcture_Trasition) + (1-alpha)*Attribute_Trasition)
    Probability = Probability + First_Probability
    Probability = normalize(Probability)
    Probability = sp.csr_matrix(Probability)
    return  Probability

def Proximity_Calculation(Attribute_Trasition,Strcture_Trasition,alpha,adj_coords):
    Probability = np.zeros_like(Strcture_Trasition)
    Probability = (1-alpha) * Attribute_Trasition
    for edge in adj_coords:
        Probability[edge[0]][edge[1]] = alpha*Strcture_Trasition[edge[0]][edge[1]] + (1-alpha) * Attribute_Trasition[edge[0]][edge[1]]
    return Probability

def cosine_similarity(x,y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        result = 0.0
    else: 
        result = num / denom
        
    return result
    