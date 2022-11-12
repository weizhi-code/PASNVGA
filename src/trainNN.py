from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import metrics
from Metrics import calculate_metrics

def generate_data(emb, posEdges, negEdges):
    #stack codings of two proteins together
    posNum = posEdges.shape[0]
    negNum = negEdges.shape[0]

    X = np.empty((posNum+negNum,2*emb.shape[1]))
    k = 0

    for x in posEdges:
        X[k] = np.hstack((emb[x[0]],emb[x[1]]))
        k = k + 1
    for x in negEdges:
        X[k] = np.hstack((emb[x[0]],emb[x[1]]))
        k = k + 1 

    Y_pos = np.full((posNum,2),[0,1])
    Y_neg = np.full((negNum,2),[1,0])
    Y = np.vstack((Y_pos,Y_neg))

    return X,Y

def train_nn(X_train, Y_train, X_test, Y_test, dataset, epoch):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=200, batch_size=128, verbose=1)

    y_prob = model.predict(X_test)
    y_classes = y_prob.argmax(axis=-1)
    y_true = Y_test[:,1]
    acc = calculate_metrics(y_true, y_classes)
    roc_score = roc_auc_score(y_true, y_prob[:,1])
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_prob[:,1])
    pr_auc = metrics.auc(recall, precision)  # 梯形块分割，建议使用

    a = open('/home/wangliwei/java/smalldata/' + dataset +'/' + str(epoch) + 'y_true.txt','w') 
    b = open('/home/wangliwei/java/smalldata/' + dataset +'/' + str(epoch) + 'y_classes.txt','w')
    c = open('/home/wangliwei/java/smalldata/' + dataset +'/' + str(epoch) + 'y_prob.txt','w')
    d = open('/home/wangliwei/java/smalldata/' + dataset +'/' + str(epoch) + 'all_y_prob.txt','w')
    for i in y_true:
        a.write (str(i) + '\n')
    for i in y_classes:
        b.write (str(i) + '\n')
    for i in y_prob[:,1]:
        c.write (str(i)+ '\n')
    for i in y_prob:
        d.write (str(i[0])+ ' ' + str(i[1]) + '\n')
    return acc,roc_score, pr_auc
    





    
