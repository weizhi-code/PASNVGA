from sklearn.metrics import roc_auc_score
from sklearn import metrics
from Metrics import calculate_metrics

def entirety(dataset):
    y_true_all = []
    y_classes_all = []
    y_prob_all = []
    for i in range(1,6):
        with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset +'/' + str(i) + 'y_prob.txt') as f:
            y_prob_1= f.readlines()
        with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset +'/' + str(i) + 'y_classes.txt') as f:
            y_classes_1 = f.readlines()
        with open('C:/Users/Administrator/Desktop/PASNVGA/dataset/' + dataset +'/' + dataset +'/' + str(i) + 'y_true.txt') as f:
            y_true_1 = f.readlines()
        for x in y_prob_1:
            y_prob_all.append(float(x))
        for y in y_classes_1:
            y_classes_all.append(int(y))
        for z in y_true_1:
            y_true_all.append(int(z))
       
    print(len(y_true_all))
    print(len(y_classes_all))
    print(len(y_prob_all))
    precision, recall, threshold = metrics.precision_recall_curve(y_true_all, y_prob_all)
    pr_auc = metrics.auc(recall, precision) 
    acc = calculate_metrics(y_true_all, y_classes_all)
    roc_score = roc_auc_score(y_true_all, y_prob_all)
      

       
    # print(y_classes_all)
    # print(y_prob_all)
    # print(y_true_all)
    print ('accuracy:',acc[0])
    print ('sensitivity:',acc[1])
    print ('specificity:',acc[2])
    print ('precision:',acc[3])
    print ('F1:',acc[4])
    print('auc:',roc_score)
    print('pr_auc:',pr_auc)
entirety('SP')