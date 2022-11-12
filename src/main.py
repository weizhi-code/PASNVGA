from statistics import product_dict
from Input import load_data
# from xulietezhen import load_data
# from wagnluotezhen import load_data
import numpy as np
import tensorflow.compat.v1 as tf
from improvement import Ehance_Matrix

tf.disable_v2_behavior()


from trainGCN import train_gcn #加了增强矩阵
from trainNN import generate_data,train_nn

def train(dataset, method,epoch,protein_num):
    adj, adj_train,  features, train_edges, train_edges_false,  test_edges, test_edges_false = load_data(dataset,method,epoch,protein_num)
    adj_part = adj_train
    adj_train = Ehance_Matrix(dataset,method,protein_num)
    #embeddings returned by VGAE
    emb = train_gcn(adj_part,features,adj_train,train_edges,train_edges_false)
    X_train,Y_train = generate_data(emb, train_edges, train_edges_false)
    X_test,Y_test = generate_data(emb, test_edges, test_edges_false)
    # 调用神经网络分类器
    acc,roc_score,pr_auc= train_nn(X_train,Y_train,X_test,Y_test,dataset,epoch)
    print ('accuracy:',acc[0])
    print ('sensitivity:',acc[1])
    print ('specificity:',acc[2])
    print ('precision:',acc[3])
    print('F1_score:',acc[4])
    print('roc:',roc_score)
    print('pr_auc:',pr_auc)

def main():
    method = 'QSOrder'
    dataset = "Human"
    protein_num = product_dict(dataset)[1]
    print(protein_num)
    epoch = 1
    train(dataset,method,epoch,protein_num)
if __name__ == "__main__":
    main()
