from __future__ import division
from __future__ import print_function

import time
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import scipy.sparse as sp
import pickle as pkl

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import OptimizerAE, OptimizerVAE
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, construct_optimizer_list

tf.compat.v1.disable_eager_execution()
def train_gcn(Adj,features,adj_train, train_edges, train_edges_false):#增加了增强矩阵
    # Settings #参数命令行解析，可以通过命令行来更改参数
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
    flags.DEFINE_integer('epochs',200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 96, 'Number of units in hidden layer 1.')
    flags.DEFINE_integer('hidden2', 48, 'Number of units in hidden layer 2.')
    flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
    flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
    flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

    model_str = FLAGS.model # 读取model参数
    mask_index = construct_optimizer_list(features.shape[0], train_edges, train_edges_false)
    adj = adj_train

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    # Define placeholderstf.compat.v1.placeholder 
    tf.compat.v1.placeholder = {
        'features': tf.sparse_placeholder(tf.float64),
        'adj': tf.sparse_placeholder(tf.float64),
        'adj_orig': tf.sparse_placeholder(tf.float64),
        'dropout': tf.placeholder_with_default(0., shape=())
    }
    
    
    # num_nodes = adj.shape[0] # 蛋白质总数
    num_nodes = Adj.shape[0] # 蛋白质总数

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1] #特征矩阵的维度
    features_nonzero = features[1].shape[0] #非零特征的个数

    # Create model 
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(tf.compat.v1.placeholder, num_features, features_nonzero)
        
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(tf.compat.v1.placeholder, num_features, num_nodes, features_nonzero)
        
    pos_weight = 1 # 计算交叉熵损失函数的权重
    norm = 1
    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(tf.compat.v1.placeholder['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          mask=mask_index)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(tf.compat.v1.placeholder['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           mask=mask_index)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    adj_label = Adj + sp.eye(Adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features, tf.compat.v1.placeholder)
        feed_dict.update({tf.compat.v1.placeholder['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.log_lik, opt.kl], feed_dict=feed_dict)
        print("Epoch:", '%04d' % (epoch+1), "train_loss=", "{:.5f}".format(outs[1]))

    print("Optimization Finished!")
    
    #return embedding for each protein
    emb = sess.run(model.z_mean,feed_dict=feed_dict)
    return emb

