# 对ppi文件进行处理，且划分成五折交叉验证所需的数据集
import numpy as np
from numpy.lib.function_base import append
from sklearn.model_selection import StratifiedKFold
import scipy.io as scio

#会返回一个蛋白质符号对应的字典，还会生产正样本文件和负样本文件
def product_dict(dataname):
    sum_dict_1 = {}
    sum_dict_2 = {}
    with open('/home/wangliwei/java/smalldata/'+dataname+'.ppi') as f: # 文件里是已知的蛋白质对是正还是负样本的标注
        data = f.readlines()
        for line in data:
            line = line.split()
            temp_list = []
            temp_dict = {}
            if line[0] in sum_dict_1:#当序列在字典中的时候
                temp_list = sum_dict_1[line[0]] # 把字典中key对应的value,放入临时数组。
                temp_list.append(line[1])# 将重复的key的value添加到临时数组
                temp_dict = {line[0] : temp_list} #生成临时字典
            else:
                temp_dict = {line[0] : [line[1]]}#序列不在字典中直接添加
            sum_dict_1.update(temp_dict) #用临时字典更新原字典
        #上面的只查重了左边的序列，下面的查重右边的序列
        for line in data:
            line = line.split()
            temp_list_1 = []
            temp_dict_1 = {}
            if line[1] in sum_dict_2:
                temp_list_1 = sum_dict_2[line[1]]
                temp_list_1.append(line[0])
                temp_dict_1 = {line[1] : temp_list_1}
            else:
                temp_dict_1 = {line[1] : [line[0]]}
            sum_dict_2.update(temp_dict_1)
        sum_dict_1.update(sum_dict_2)#将两个字典一起更新，得到所有不同的序列
        num = len(sum_dict_1)
        #给序列对应上数字从1开始
        num_dict = {}
        i = 1
        for key in sum_dict_1.keys():
            Temp_dict = {key : i}
            i = i + 1               
            num_dict.update(Temp_dict)

        return num_dict, num

def seq_num(dataname):
    with open('/home/wangliwei/java/smalldata/'+dataname+'-2.ppi') as f:
        data = f.readlines()
    a = product_dict(dataname)
    e = open('/home/wangliwei/java/smalldata/' + dataname +'/pos.txt','w')
    f = open('/home/wangliwei/java/smalldata/' + dataname +'/neg.txt','w')
    g = open('/home/wangliwei/java/smalldata/'+dataname+'.ppi','w')
    for line in data:
        x,y,z = line.split()
        # Human下面这些序列不对应序列编码
        if (dataname == 'Human'):
            if (x != 'NP_116199.2' and x != 'NP_060098.3') and (y != 'NP_116199.2' and y != 'NP_060098.3'):
                g.write(x + " " + y +" " + z +"\n")
                x = a[0][x]
                y = a[0][y]
                if z == '1':#分正负蛋白质对，正为1负为-1
                    e.write(str(x)+"::"+ str(y) + "::"+ str(1)+ "\n")
                else:
                    f.write(str(x)+"::"+ str(y) + "::"+ str(-1)+ "\n")
        elif(dataname == 'yeast'):
            x = a[x]
            y = a[y]
            if z == '1':
                e.write(str(x)+"::"+ str(y) + "::"+ str(1)+ "\n")
            else:
                f.write(str(x)+"::"+ str(y) + "::"+ str(-1)+ "\n")
        else:
            x = a[x]
            y = a[y]
            if z == '+1.0':
                e.write(str(x)+"::"+ str(y) + "::"+ str(1)+ "\n")
            else:
                f.write(str(x)+"::"+ str(y) + "::"+ str(-1)+ "\n")
                

        

def datasplit(name):
  
    with open('/home/wangliwei/java/smalldata/' + name +'/pos.txt') as f:
        data = f.readlines()
    all_edge = []
    for line in data:
        all_edge.append(line)
    np.random.shuffle(all_edge)#随机打乱
    all_edge = np.array(all_edge)
    y = np.ones(len(all_edge))
    skf = StratifiedKFold(n_splits=5).split(all_edge,y)#五折交叉验证
    i = 1
    for train_index, test_index in skf:
        a = open('/home/wangliwei/java/smalldata/' + name + '/' + str(i) + 'pos_'+ name +'_train.txt','w')
        b = open('/home/wangliwei/java/smalldata/' + name + '/'+ str(i) + 'pos_'+ name +'_test.txt','w')
        i = i +1
        X_train, X_test = all_edge[train_index], all_edge[test_index]
        for x in X_train: 
            a.write(x)
        for y in X_test:
            b.write(y)
    with open('/home/wangliwei/java/smalldata/' + name +'/neg.txt') as f:
        negdata = f.readlines()
    all_edge_neg = []
    for line in negdata:
        all_edge_neg.append(line)
    np.random.shuffle(all_edge_neg)#随机打乱
    all_edge_neg = np.array(all_edge_neg)
    y = np.ones(len(all_edge_neg))
    skf = StratifiedKFold(n_splits=5).split(all_edge_neg,y)
    n = 1
    for train_index_neg, test_index_neg in skf:
        a = open('/home/wangliwei/java/smalldata/'+ name +'/'+ str(n) +  'neg_'+ name +'_train.txt','w')
        b = open('/home/wangliwei/java/smalldata/'+ name +'/'+ str(n) +  'neg_'+ name +'_test.txt','w')
        n = n + 1
        neg_train, neg_test = all_edge_neg[train_index_neg], all_edge_neg[test_index_neg]
        for x in neg_train:  
            a.write(x)
        for y in neg_test:
            b.write(y)


def format (name):
    with open('/home/wangliwei/java/smalldata/' + name +'/pos1.txt') as f:
        posdata = f.readlines()
    with open('/home/wangliwei/java/smalldata/' + name +'/neg1.txt') as f:
        negdata = f.readlines()
    c = open('/home/wangliwei/java/smalldata/' + name +'/pos.txt','w')
    d = open('/home/wangliwei/java/smalldata/' + name +'/neg.txt','w')
    for line in posdata:
        pos_row = line.split()
        pos_row [0]= str(int(pos_row[0]) + 1)
        pos_row [1]= str(int(pos_row[1]) + 1)
        # print(pos_row)
        c.write(pos_row[0]+"::"+ pos_row[1] + "::" + '1' + "\n")
    for line in negdata:
        neg_row = line.split()
        neg_row [0]= str(int(neg_row[0]) + 1)
        neg_row [1]= str(int(neg_row[1]) + 1)
        d.write(neg_row[0]+"::"+ neg_row[1] + "::" + '-1' + "\n")
    
    




if __name__ == "__main__":
    # a= product_dict('Human')[1]
    # print(a)
#     for i in ['AT','EC','SP','Human','yeast']:
    seq_num('Human')
#         datasplit('i')

#     for i in ['human_57','yeast_57', 'H.pylori_57']:
#         seq_num(i)
#         datasplit('H.pylori')
    

    
