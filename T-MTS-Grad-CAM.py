# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:13:31 2020
由于沿特征方向的结果较差，且沿特征方向的排列缺乏依据，所以暂不考虑！
@author: ZhuWen
"""

from tensorflow.keras.models import Model,load_model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numb_classes = 3
Class_index = 0
Uniform_class = 0.1
Var = 18

mts_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\x_test.npy'
Label_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\y_test.npy'
Indexlabel_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\test_index.npy'

T_model_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\T_best_model.hdf5'
T_pred_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\T_y_pred.npy'
T_Layer_name = 'activation_8'#层的名称有待进一步确定

V_model_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\V_best_model.hdf5'
V_pred_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试\V_y_pred.npy'

V_Layer_name =[]
for i in range(2,56,3):#先查询再确定
    V_Layer_name.append('activation_%s'%(i))

save_path = r'F:\小论文\盾构论文\地层损失状态预测\4\可视化测试'


def Scan_result(Indexlabelpath, Labelpath, predpath, numbclasses, Classindex):#此处要输出标签和预测值以及分数，首先确保预测正确，其次分数最高，保证呈现质量
    
    Indexlabel = np.load(Indexlabelpath)#Indexlabel输入的时候是1维一行的数组
    label = np.load(Labelpath)#label输入的时候是2维一行的数组
    pred = np.load(predpath)
    
    Lscore = [[] for i in range(numbclasses)]#这里不能图省事将空集划等号
    
    TPFN_index = [[] for i in range(numbclasses)]#TP+FN
    TPFP_index = [[] for i in range(numbclasses)]#TP+FP
    TP_index = [[] for i in range(numbclasses)]
    
    for j in range(label.shape[0]):
        if label[j][0] == np.argmax(pred[j]):
            Lscore[int(label[j][0])].append(max(pred[j]))
            TP_index[int(label[j][0])].append(j)#same的意思就是两个方向同时预测正确
        for k in range(numbclasses):
            if np.argmax(pred[j]) == k:
                TPFP_index[k].append(j)#预测为正的样本序号
            if label[j][0] == k:
                TPFN_index[k].append(j)#实际为正的样本序号
    #此处用两个if是因为两者之间是并列关系，不是else关系，else满足第一个会跳过余下的，注意逻辑!!!这是复选！！！！！！
    for n in range(numbclasses):
        print('第%d类的presion：%f'%(n, (len(TP_index[n])/len(TPFP_index[n]))))
        print("第%d类的recall：%f"%(n, (len(TP_index[n])/len(TPFN_index[n]))))
        print("第%d类的F1：%f"%(n, 2*(len(TP_index[n])/len(TPFN_index[n]))*(len(TP_index[n])/len(TPFP_index[n]))\
                            /((len(TP_index[n])/len(TPFN_index[n]))+(len(TP_index[n])/len(TPFP_index[n])))))
        print("第%d类的最优样本序号：%d！在原始样本中对应的序号为：%d"%(n,\
                                                TP_index[n][np.argmax(Lscore[n])], Indexlabel[TP_index[n][np.argmax(Lscore[n])]]))
    
    return TP_index[Classindex][np.argmax(Lscore[Classindex])],2*(len(TP_index[Classindex])/len(TPFN_index[Classindex]))*(len(TP_index[Classindex])/len(TPFP_index[Classindex]))\
                            /((len(TP_index[Classindex])/len(TPFN_index[Classindex]))+(len(TP_index[Classindex])/len(TPFP_index[Classindex])))#返回该类别预测准确且分数最高的序号

def Uniform_weight_1D(D, Uniformclass):
    temp = np.zeros((len(D),))
    for i in range(len(D)):
        for j in np.arange(0.1,1.1,Uniformclass):
            if D[i] == 0:
                temp[i] = D[i]
            elif j-Uniformclass < D[i] <= j:
                temp[i] = format(j-Uniformclass, '.1f')#np.arange()可能有很多小数，format可以四舍五入
            else:
                continue
    return temp

def Uniform_weight_2D(D, Uniformclass):
    temp = np.zeros((D.shape[0],D.shape[1]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            for k in np.arange(0.1,1.1,Uniformclass):
                if D[i][j] == 0:
                    temp[i][j] = D[i][j]
                elif k - Uniformclass < D[i][j] <= k:
                    temp[i][j] = format(k, '.1f')#np.arange()可能有很多小数，format可以四舍五入
                else:
                    continue
    return temp

def Heatmap(FCAM, FSCORE,TCAM, TSCORE):
    FT_cam = np.zeros((len(FCAM),len(TCAM)))
    for i in range(len(FCAM)):
        for j in range(len(TCAM)):
            FT_cam[i][j] =  TCAM[j]+FCAM[i]*FSCORE/TSCORE#此处还是用相加的好，权重相加容易理解，小于1的数相乘会缩小比例
    heatmap = norm(FT_cam)#将数值变换到（0,1）之间，且负值通通根据最小值定义
    
    return heatmap
#########################################################################################
def prepare_input(x):
    new_x = []
    n_vars = x.shape[2]
    
    for i in range(n_vars):
        new_x.append(x[:,:,i:i+1])
    
    return  new_x

def loading(mtspath, modelpath, index):
    mts = np.load(mtspath)[index]#序号数字需要重新确定，分析第几类，输入第几类的数据，该数据可能需要重新确定，选取正确预测且效果好的
    mts = np.expand_dims(mts, axis=0)
    model = load_model(modelpath)
    model.summary()
    return mts, model

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):#权重的归一化非常重要！！！在加权之前！！！L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-20)

def norm(y):#权重的归一化非常重要！！！在加权之前！！！L2 norm
    return (y - np.min(y))/((np.max(y) - np.min(y)) + 1e-20)


class GradCAM_T:
    def __init__(self, model, classIdx, layerName):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        
    def compute_heatmap(self, mts):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(mts, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            Score = predictions[:, self.classIdx]#此处的loss计算与Jacobgil-Grad-CAM是不一样的，
                                                   #因为tf的版本不同，但是同样为分数的表达应该是一样的
            print(Score)
        grads = normalize(tape.gradient(Score, convOutputs))
        #print(grads)
        weights = np.mean(grads[0], axis = 0)
        cam = np.ones(convOutputs[0].shape[0], dtype = np.float32)
        
        for i, w in enumerate(weights):
            cam += w * convOutputs[0][:, i] #按列加权，按行相加
            
        cam = np.maximum(cam, 0)#此处没有使用ReLU函数，只是将负值赋值为0并缩放，相当于ReLU函数
        cam = cam / np.max(cam)
    
        return cam, Score

class GradCAM_V:
    def __init__(self, model, classIdx, layerName):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        
    def compute_heatmap(self, mts,Var):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        #gradModel.summary()
        with tf.GradientTape() as tape:
            inputs = tf.cast(mts, tf.float32)
            INPUTS = []
            for n in range(Var):
                INPUTS.append(inputs[n])
            (convOutputs, predictions) = gradModel(INPUTS)
            Score = predictions[:, self.classIdx]#此处的loss计算与Jacobgil-Grad-CAM是不一样的，
                                                   #因为tf的版本不同，但是同样为分数的表达应该是一样的
            #print(Score)
        grads = normalize(tape.gradient(Score, convOutputs))
        #print(grads)
        weights = np.mean(grads[0], axis = 0)
        cam = np.ones(convOutputs[0].shape[0], dtype = np.float32)
        
        for i, w in enumerate(weights):
            cam += w * convOutputs[0][:, i] #按列加权，按行相加
            
        cam = np.maximum(cam, 0)#此处没有使用ReLU函数，只是将负值赋值为0并缩放，相当于ReLU函数
        cam = cam / np.max(cam)
    
        return cam, Score

def T_Grad_CAM(mtspath, modelpath, index, Layername):
    MTS, model = loading(mtspath, modelpath, index)
    pre = model.predict(MTS)
    pre_class = np.argmax(pre)
    T_cam, Tscore = GradCAM_T(model, pre_class, Layername).compute_heatmap(MTS)
    print('测试样本为第%s类'%(pre_class))
    print('测试样本为第%s类的得分%s'%(pre_class,Tscore))
    Tcam=norm(T_cam)
    return Tcam, Tscore

def V_Grad_CAM(mtspath, modelpath, index, Layername,Var):
    MTS, model = loading(mtspath, modelpath, index)
    MTS = MTS.transpose(0,2,1)
    MTS = prepare_input(MTS)
    #print(MTS.shape)
    pre = model.predict(MTS)
    pre_class = np.argmax(pre)
    print(pre_class)
    V_cam = np.zeros((Var,))
    for i in range(Var):
        cam, Vscore = GradCAM_V(model, pre_class, Layername[i]).compute_heatmap(MTS,Var)
        V_cam[i]=np.mean(cam)
    Vcam=norm(V_cam)
    print('测试样本为第%s类'%(pre_class))
    print('测试样本为第%s类的得分%s'%(pre_class,Vscore))
    return Vcam, Vscore

##################################################第一部分求沿“时间轴”的权重分布
T_ID,T_F1 = Scan_result(Indexlabel_path, Label_path, T_pred_path, numb_classes, Class_index)#ID是label中属于该类分类效果最好的index
V_ID,V_F1 = Scan_result(Indexlabel_path, Label_path, V_pred_path, numb_classes, Class_index)


TCAM, TSCORE = T_Grad_CAM(mts_path, T_model_path, T_ID, T_Layer_name)#分两段此处确定要查看的类别
VCAM, VSCORE = V_Grad_CAM(mts_path, V_model_path, V_ID, V_Layer_name,Var)


HEAT = Heatmap(TCAM,T_F1,VCAM,V_F1)

P_MTS = np.load(mts_path)[0]
##################################################
T_CAM = pd.DataFrame(TCAM)#时间轴方向的权重分布
T_CAM.to_csv(save_path + '时间轴方向.csv', header=None)

V_CAM = pd.DataFrame(VCAM)#变量轴方向的权重分布
V_CAM.to_csv(save_path + '特征变量方向.csv', header=None)

HM = pd.DataFrame(HEAT)#热力图
HM.to_csv(save_path + 'Heatmap.csv', header=None)

P_MTS = pd.DataFrame(P_MTS)#样例数据
P_MTS.to_csv(save_path + 'TEST_MTS.csv', header=None)

##########################################################由于原始结果的权重“驳杂”，难以区分，因此此处根据区间划分将权重统一
'''
U_CAM = Uniform_weight_1D(CAM, Uniform_class)
T_U_CAM = pd.DataFrame(U_CAM)#时间轴方向的权重分布
T_U_CAM.to_csv(save_path + 'T_U_CAM.csv', header=None)
'''

