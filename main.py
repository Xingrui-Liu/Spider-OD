# -*- coding: utf-8 -*-
import SpiderODv5
import scipy.io as scio
from pandas.core.frame import DataFrame
import time
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
import os
import gc
import matplotlib.pyplot as plt
import notifyemail
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.rod import ROD
from pyod.models.suod import SUOD
from sklearn import preprocessing


def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


def save_report_row(y_true, y_predict_type, y_predict_scores):
    report_row = []
    # accuracy_score
    report_row.append(accuracy_score(y_true, y_predict_type))
    # metrics
    report_row.append(metrics.precision_score(y_true, y_predict_type, average='micro'))
    report_row.append(metrics.precision_score(y_true, y_predict_type, average='macro'))
    # recall
    report_row.append(metrics.recall_score(y_true, y_predict_type, average='micro'))
    report_row.append(metrics.recall_score(y_true, y_predict_type, average='macro'))
    # F1
    report_row.append(metrics.f1_score(y_true, y_predict_type, average='weighted'))
    # kappa score
    report_row.append(cohen_kappa_score(y_true, y_predict_type))
    # ROC
    report_row.append(roc_auc_score(y_true, y_predict_scores))
    # 距离
    # 海明距离
    report_row.append(hamming_loss(y_true, y_predict_type))
    # Jaccard距离
    report_row.append(jaccard_score(y_true, y_predict_type))
    # 回归
    # 可释方差值（Explained variance score）
    report_row.append(explained_variance_score(y_true, y_predict_type))
    # 平均绝对误差（Mean absolute error）
    report_row.append(mean_absolute_error(y_true, y_predict_type))
    # 均方误差（Mean squared error）
    report_row.append(mean_squared_error(y_true, y_predict_type))
    # 中值绝对误差（Median absolute error）
    report_row.append(median_absolute_error(y_true, y_predict_type))
    # R方值，确定系数
    report_row.append(r2_score(y_true, y_predict_type))
    return report_row


def randomData(mu, sigma, row, col):
    """
    Production of Gaussian distribution data
    :param mu: mean
    :param sigma: standard deviation
    :param row: number of rows
    :param col: number of columns
    :return: Gaussian distribution data set
    """
    data1 = np.random.normal(mu, sigma, [row, col])
    data2 = np.random.normal(mu/2, sigma/2, [row, col])
    data3 = np.append(data1, data2, axis=0)
    return data1


def normalize(list, value):
    range = max(list) - min(list)
    if range == 0:
        return 1
    else:
        value2 = (value - min(list)) / range
        return value2


# 决策图
def scatterPlotPhe(data, pheList):
    """
    Data visualization ( pheromone map )
    :param data: Data set
    :return: No return
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()

    for i in range(0, len(pheList)):
        now_pheList = normalize(pheList, pheList[i])
        plt.scatter(x[i], y[i], marker=".", alpha=0.3, s=now_pheList * 1000, c='orange')
        # plt.scatter(x[i], y[i], marker=".",alpha=0.3,s=100, c='orange')

        ax.annotate(round(pheList[i], 2), (x[i], y[i]))
    # plt.savefig('img\phefig.svg', dpi=600)
    plt.show()


# 点图像
def scatterPlot(data):
    """
    Data visualization ( scatter plot )
    :param data: data set
    :return: No return
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, txt in enumerate(n):
        ax.annotate(i+1, (x[i], y[i]))

    str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    plt.savefig("res_img/" + str + ".svg", dpi=600)
    plt.show()


email_flag=0
# --------邮件---------
if email_flag == 1:
    receivers_list = ["xxxxxxxxxx@qq.com"]    # 【修改】收件人邮箱，可以添加多个，用逗号分割
    notifyemail.Reboost(mail_host='smtp.163.com', mail_user='xxxxxxxxxxx@163.com', mail_pass='xxxxxxxxxxxxxxxx',
                            default_reciving_list=receivers_list, log_root_path='logs', max_log_cnt=5)
    description = ""  # 【可修改】对于这次实验的描述（比如描述模型的结构、数据集等）
# --x------邮件-------x--

datasets_list = ["glass"]     # TODO 修改1：数据集
# "glass","ecoli","ionoSphere","vowels"
# glass 176-184
# glass 0.042; ecoli 0.026 Ionophere 0.36 Vowels 0.034


algorithms_list = ["Spider"]     # TODO 修改2：算法名    "Spider"
# "KNN","LOF","COPOD","ECOD","HBOS","OCSVM","ROD","SUOD"
for now_dataset in range(0, len(datasets_list)):
    print("now_dataset= ", datasets_list[now_dataset])
    for now_algorithm in range(0, len(algorithms_list)):
        dataName = datasets_list[now_dataset]
        algorithm = algorithms_list[now_algorithm]
        if dataName != "ecoli":
            dataFile = "datasets/" + dataName + ".mat"
            input_data = scio.loadmat(dataFile)
            X_train = input_data['X']
            y_true = input_data['y']
        else:
            input_data = np.loadtxt("datasets/ecoli.csv", delimiter=",")
            X_train = input_data[:, 0:6]
            y_true = input_data[:, 7]
        pointNum = np.shape(X_train)[0]
        outlier_num = sum(y_true)

        # 运行代码
        # TODO 修改3：k值 注意左闭右开
        start_k = 8
        end_k = 91

        print("now_algorithm= ",algorithms_list[now_algorithm])
        for k in range(start_k, end_k):
            gc.collect()
            print("k=", k)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            start_time = time.perf_counter()
            info = []
            report = []
            # TODO pyod调用以下代码
            # 按k近邻执行
            now_contamination = 0.0556
            detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                             LOF(n_neighbors=25), LOF(n_neighbors=35)]
            if algorithm == "LOF":
                clf = LOF(n_neighbors=k, contamination=now_contamination)   # n_neighbors=20, algorithm='auto', leaf_size=30,metric='minkowski', p=2, metric_params=None,contamination=0.1, n_jobs=1, novelty=True
            elif algorithm == "SOD":
                clf = SOD(n_neighbors=k, ref_set=2,contamination=now_contamination)    # contamination=0.1, n_neighbors=20, ref_set=10, alpha=0.8
            elif algorithm == "KNN":
                clf = KNN(n_neighbors=k, contamination=now_contamination)    # contamination=0.1, n_neighbors=5, method='largest',radius=1.0, algorithm='auto', leaf_size=30,metric='minkowski', p=2, metric_params=None, n_jobs=1,**kwargs
            elif algorithm == "HBOS":
                clf= HBOS(n_bins=k, contamination=now_contamination)  # n_bins=10, alpha=0.1, tol=0.5, contamination=0.1
            elif algorithm == "SUOD":
                clf = SUOD(contamination=k*0.01, n_jobs=1)
            elif algorithm == "COPOD":
                clf = COPOD(contamination=k*0.01)  # contamination=0.1, n_jobs=1
            elif algorithm == "ECOD":
                clf = ECOD(contamination=k*0.01)  # contamination=0.1, n_jobs=1
            elif algorithm == "OCSVM":
                clf = OCSVM(nu=k*0.01, contamination=now_contamination)  # kernel='rbf', degree=3, gamma='auto',coef0=0.0,tol=1e-3, nu=0.5, shrinking=True, cache_size=200,verbose=False, max_iter=-1, contamination=0.1


            # TODO spider代码
            elif algorithm == "Spider":
                # # TODO 数据集归一化处理
                standard_scaler = preprocessing.StandardScaler()
                standard_scaler.fit(X_train)
                standard_scaler_data = standard_scaler.transform(X_train)

                contamination = 0.15
                # glass 0.042; ecoli 0.026 Ionosphere 0.36 Vowels 0.034
                # contamination = 0.01 * k  TODO  0.5/200*k
                # max_k = int(X_train.shape[0] * contamination)
                # TODO standard_scaler_data X_train
                y_predict_scores, y_predict_type, indices = SpiderODv5.Model(standard_scaler_data, T=20, energy=5, alpha=0.95,
                                                            beta=0.8, gamma=0.8, contamination=contamination, Dthr=k)
            #      T=50, energy=5, alpha=0.95, beta=0.8, gamma=0.8, contamination=0.3, Dthr=10
            # TODO pyod调用以下代码
            else:
                clf.fit(X_train)  # 使用X_train训练检测器clf
                # #
                # # 返回训练数据X_train上的异常标签和异常分值
                y_predict_type = clf.labels_  # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值)
                y_predict_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)
                y_predict_confidence = [-1] * pointNum
                end_time = time.perf_counter()

            # save result
            # # TODO 测试无穷大和空值
            # print(np.isinf(y_true))
            # print(np.isnan(y_true))
            # print(np.isinf(y_predict_type))
            # print(np.isnan(y_predict_type))
            # print(np.isinf(y_predict_scores))
            # print(np.isnan(y_predict_scores))
            # TODO 针对出现无穷大和空值的修改
            # max_value = np.finfo(np.float64).max
            # y_predict_scores = np.where(np.isinf(y_predict_scores), max_value, y_predict_scores)
            # y_predict_scores = np.nan_to_num(y_predict_scores, nan=0.0)

            report_row = save_report_row(y_true, y_predict_type, y_predict_scores)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            save_row = [dataName, algorithm, k, run_time]
            save_row.extend(report_row)
            report.append(save_row)
            y_predict_confidence = [-1] * pointNum
            rownames = np.array(range(1, np.size(y_predict_scores, 0) + 1)).astype(int)  # 点号
            socore_data = np.c_[rownames.T, np.array(y_predict_type).T, np.array(y_predict_scores).T, np.array(y_predict_confidence).T]

            # ----------保存每轮的数据---------
            # 点号；预测类型；预测得分；预测置信度
            if not os.path.exists("scoreData/" + dataName + "/" + algorithm):
                os.makedirs("scoreData/" + dataName + "/" + algorithm)
            print("------save scoreData------")
            np.savetxt("scoreData/" + dataName + "/" + algorithm + "/" + str(k) + ".csv", socore_data, delimiter=',',
                       comments='', header="pointNum,type,score,confidence")

            # 混淆矩阵
            if not os.path.exists("confusionMatrix/" + dataName + "/" + algorithm):
                os.makedirs("confusionMatrix/" + dataName + "/" + algorithm)
            print("------save confusionMatrix------")
            np.savetxt("confusionMatrix/" + dataName + "/" + algorithm + "/" + str(k) + ".csv",
                       confusion_matrix(y_true, y_predict_type), delimiter=',', comments='', header="0,1")
            # --x--------保存每轮的数据-------x--

            # ----------保存所有轮的数据----------
            report_dataFrame = DataFrame(report)
            report_dataFrame.columns = ["dataName", "algorithm", "k", "run_time","accuracy_score", "metrics_micro",
                                        "metrics_macro", "recall_micro", "recall_macro", "F1", "kappa_score", "ROC",
                                        "Hamming_distance", "Jaccard_distance", "Explained_variance_score",
                                        "Mean_absolute_error",
                                        "Mean_squared_error", "Median_absolute_error", "r2_score"]

            if not os.path.exists("Report/" + dataName + "/" + algorithm):
                os.makedirs("Report/" + dataName + "/" + algorithm)
                print("------Make new dir------")
            if not os.path.exists("Report/" + dataName + "/" + algorithm + "/" + str(start_k) + "_" + str(end_k) + ".csv"):
                with open("Report/" + dataName + "/" + algorithm + "/" + str(start_k) + "_" + str(end_k) + ".csv", 'ab') as f:
                    np.savetxt(f, [], delimiter=',', fmt='%s', comments='',header=",".join(report_dataFrame.columns))
                           # header="dataName, algorithm, k, run_time, precision_score_mac,precision_score_mic, precision_score_weight, precision_score_none,accuracy_score, metrics_micro,metrics_macro, recall_micro, recall_macro, F1, kappa_score, ROC,Hamming_distance, Jaccard_distance, Explained_variance_score, Mean_absolute_error,Mean_squared_error, Median_absolute_error, r2_score")
                # --x--------保存所有轮的数据--------x--
            print("------save Report------")
            with open("Report/" + dataName + "/" + algorithm + "/" + str(start_k) + "_" + str(end_k) + ".csv", 'ab') as f:
                np.savetxt(f, report_dataFrame, delimiter=',', fmt='%s', comments='')
            # --x--------保存所有轮的数据--------x--dat


if email_flag == 1:
    notifyemail.add_text("使用数据集: "+str(datasets_list))
    notifyemail.add_text("使用算法: "+str(algorithms_list))
    notifyemail.add_text(description)
    notifyemail.add_file("./"+"Report")
    notifyemail.add_file("./"+"scoreData")
    notifyemail.add_file("./"+"datasets")
    notifyemail.add_file("./"+"confusionMatrix")
    notifyemail.send_log()
    # os.system('shutdown -s -t 1')  # 1代表一秒内关机，可自行设置