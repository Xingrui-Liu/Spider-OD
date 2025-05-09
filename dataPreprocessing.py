import matplotlib.pyplot as plt
import numpy as np  # 调用 numpy 包作为 np 使用
from scipy.spatial import distance  # 调用distance函数求距离矩阵
import seaborn as sns
import pandas as pd
from scipy.io import arff


def loadData(filename):
    """
    加载数据
    :param filename: 文件名
    :return: 数据集 numpy集合的形式
    """
    data = np.loadtxt(filename)  # 加载数据
    return data


def loadData2(filename):
    """
    加载数据
    :param filename: 文件名
    :return: 数据集 numpy集合的形式
    """
    data = np.loadtxt(filename, delimiter=',')  # 加载数据
    return data


def scatterPlot(data):
    """
    数据可视化（散点图）
    :param data: 数据集
    :return: 无返回
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt + 1, (x[i], y[i]))
    plt.savefig('img\meta_data.svg', dpi=600)
    plt.show()


def scatterPlotPhe(data, pheList):
    """
    数据可视化（散点图）
    :param data: 数据集
    :return: 无返回
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()

    for i in range(0, len(pheList)):
        now_pheList = normalize(pheList, pheList[i])
        plt.scatter(x[i], y[i], marker=".", s=now_pheList * 1000, c='orange')
        ax.annotate(i + 1, (x[i], y[i]))
    # plt.savefig('img\phefig.svg', dpi=600)
    plt.show()

def scatterPlotLine(data, num,history_run_list):
    """
    数据可视化（散点图）
    :param data: 数据集
    :return: 无返回
    """

    x = data[:, 0]
    y = data[:, 1]
    fig, ax = plt.subplots()
    x_list=[]
    y_list=[]
    # for i in range(0, len(pheList)):
    #     now_pheList = normalize(pheList, pheList[i])
    #     plt.scatter(x[i], y[i], marker=".", s=now_pheList * 1000, c='orange')
    #     ax.annotate(i + 1, (x[i], y[i]))
    for j in range(0,len(history_run_list)):
        now_point=history_run_list[j]-1
        now_x=x[now_point]
        now_y=y[now_point]
        x_list.append(now_x)
        y_list.append(now_y)
    ax.plot(x_list, y_list, color='black', label='.', marker='.',markersize=10,linewidth=1,alpha=0.3,markerfacecolor='red')
    figname='img\line_'+str(num)+'.svg'
    plt.savefig(figname, dpi=600)
    plt.show()


def distanceMatrix(matrix):
    """
    距离矩阵
    :param matrix: 原始坐标数据构成的矩阵
    :return: dis_matrix: 距离矩阵
    """
    matrix = np.array(matrix, dtype=np.float64)  # 把传入的matrix转化为numpy类型的矩阵( ndarray )
    dis_matrix = distance.cdist(matrix, matrix,
                                'euclidean')  # 调用distance函数，求矩阵AB的距离，A=matrix，B=matrix，‘Euclidean’表示欧式距离
    return dis_matrix  # 返回一个距离矩阵


def neighborPointList(p1, disMatrix):
    """
    得到近邻矩阵里的p1的那一列，依次排列出p1的近邻
    :param p1:要找近邻的点
    :param disMatrix:距离矩阵
    :return:按序排列的p1的近邻
    """
    keys = []  # 创建list
    for i in range(1, disMatrix.shape[1] + 1):  # 创建key值
        keys.append(i)
    a = dict(zip(keys, disMatrix[p1 - 1]))  # 压缩字典
    a = sorted(a.items(), key=lambda x: x[1])  # 更新排序后为列表
    neighborArr = []  # p1近邻矩阵的那列

    for i in range(1, len(a)):
        neighborArr.append(a[i][0])  # 把返回的值加入的近邻列
    return neighborArr


def neighborListALL(disMatrix):
    """
    得到完整的近邻矩阵
    :param disMatrix:距离矩阵
    :return:近邻矩阵
    """
    res = []
    for i in range(0, disMatrix.shape[1]):  # 把近邻列加入到近邻矩阵内
        sorted_id = sorted(range(len(disMatrix[i])), key=lambda k: disMatrix[i][k], reverse=False)
        temp = sorted_id
        sorted_id = sorted(range(len(temp)), key=lambda k: temp[k], reverse=False)
        res.append(sorted_id)
    return np.array(res)


def pheromonesList(data):
    pointNum = data.shape[0]  # 求点的数量
    pheList = []  # 建立信息素列表
    for i in range(0, pointNum):
        pheList.append(1)  # 往信息素列表中添加值
    return pheList


def randomData(mu, sigma, row, col):
    """
    产生高斯分布数据
    :param mu: 均值
    :param sigma: 标准差
    :param row: 行数
    :param col: 列数
    :return: 高斯分布数据集
    """
    data1=np.random.normal(mu, sigma, [row, col])
    data2=np.random.normal(mu/2, sigma/2, [row, col])
    data3=np.append(data1,data2,axis=0)
    return data3


def normalize(list, value):
    range = max(list) - min(list)
    if range == 0:
        return 1
    else:
        value2 = (value - min(list)) / range
        return value2


def getReciprocal(matrix):
    """
    获得矩阵的倒数
    :param matrix:传入矩阵
    :return: 矩阵的倒数
    """

    return np.divide(1, matrix, out=np.zeros_like(matrix, np.float64), where=matrix != 0)


def getNND(neighborMatrix):
    """
    获得矩阵的NND
    :param matrix:传入矩阵
    :return: 返回NND列表
    """
    NND = []  # NND列表
    # i j 表示点的序号，所以从1开始
    for i in range(0, neighborMatrix.shape[0]):
        NND.append(np.mean(np.divide(neighborMatrix[i], neighborMatrix.T[i], out=np.zeros_like(neighborMatrix[i], np.float64), where=neighborMatrix.T[i] != 0)))
    return NND


def getDelta(p1, p2, disMatrix):
    """
    得到Δ([p1,k],p2]) 在k个近邻内，以p1为主视角，p2是p1的第几近邻
    :param p1: 第一个点
    :param p2: 第二个点
    :param k: k近邻
    :param disMatrix: 距离矩阵
    :return: Δ([p1,k],p2])
    """
    keys = []
    for i in range(1, disMatrix.shape[1] + 1):  # 创建key值
        keys.append(i)
    a = dict(zip(keys, disMatrix[p1 - 1]))  # 压缩字典
    a = sorted(a.items(), key=lambda x: x[1])  # 更新排序后为列表
    delta = 0
    for i in range(1, len(a)):  # 从更新后的排序里找到p2的位置
        if a[i][0] == p2:
            delta = i
    return delta + 1  # 这里假设 自己和自己 是第一近邻


def getNabla(p1, p2, disMatrix):
    """
    得到Δ([p1,k],p2]) 在k个近邻内，以p1为主视角，p2是p1的第几近邻
    :param p1: 第一个点
    :param p2: 第二个点
    :param k: k近邻
    :param disMatrix: 距离矩阵
    :return: Δ([p1,k],p2])
    """
    return getDelta(p2, p1, disMatrix)  # 把主视角改为p2就求了p1的nabla


def gen_clusters():
    """
    生成多个簇
    :return:多个簇的数据集
    """
    mean1 = [0, 5]
    cov1 = [[1, 0], [0, 10]]
    data = np.random.multivariate_normal(mean1, cov1, 10)

    mean2 = [10, 10]
    cov2 = [[10, 0], [0, 1]]
    data = np.append(data,
                     np.random.multivariate_normal(mean2, cov2, 15),
                     0)

    mean3 = [10, 20]
    cov3 = [[3, 0], [0, 4]]
    data = np.append(data,
                     np.random.multivariate_normal(mean3, cov3, 20),
                     0)

    return np.round(data, 4)


def save_data(data, filename):
    with open(filename, 'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i, 0]) + ',' + str(data[i, 1]) + '\n')


def getTypeList(data):
    """
    获取TypeList，用于表示数据的类型
    :param data:数据矩阵
    :return: TypeList
    """
    TypeList = []
    for i in range(0, data.shape[0]):  # 从0开始赋值。 typeList[0]就是第一个点的类
        TypeList.append(-1)
    return TypeList


def resPlot(data, typeList):
    """
    作出聚类图（仅适用于二维数据）
    :param data:数据
    :param typeList:类型
    :return:
    """
    df1 = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})  # 获得X、Y对应数据
    df2 = pd.DataFrame(typeList)  # 获得类型
    df1.insert(df1.shape[1], 'type', df2)
    sns.lmplot(x='x', y='y', hue='type',
               data=df1, fit_reg=False)

    return plt


def getMarkList(data):
    """
    获得每个点被划分为不同种类的次数。markList[0]-->[1,1,2,3,3,3]说明在爬行过程中1号点被划分为type3有3次，type2有2次，type1有1次
    :param data:
    :return:
    """
    pointNum = data.shape[0]  # 获得点数目
    markList = [[]] * pointNum  # 为每个点创造一个列表，用以记录被划分为什么类型
    return markList


def arffRead(url):
    data = arff.loadarff(url)
    df = pd.DataFrame(data[0])
    return df


def GridPlot(data, typeList,grid_list,grid_length):
    """
    作出聚类图（仅适用于二维数据）
    :param data:数据
    :param typeList:类型
    :return:
    """
    df1 = pd.DataFrame({'x': data[:, 0], 'y': data[:, 1]})  # 获得X、Y对应数据
    df2 = pd.DataFrame(typeList)  # 获得类型
    df1.insert(df1.shape[1], 'type', df2)
    sns.lmplot(x='x', y='y', hue='type',
               data=df1, fit_reg=False)

    for i in range(0,len(grid_list)):
        #line1
        x0=grid_list[i][0]*grid_length
        y0=grid_list[i][1]*grid_length
        x1=(grid_list[i][0]+1)*grid_length
        y1=(grid_list[i][1]+1)*grid_length

        plt.hlines(y0,x0,x1,colors="red")
        plt.hlines(y1,x0,x1,colors="red")
        plt.vlines(x0,y0,y1,color="red")
        plt.vlines(x1,y0,y1,color="red")
        # plt.plot([x0,y0],[x0,y1],color="red")
        # plt.plot([x0,y0],[x1,y0],color="red")
        # plt.plot([x1,y0],[x1,y1],color="red")
        # plt.plot([x0,y1],[x1,y1],color="red")

    return plt

def DecisionPlot(x,y):
    """
    数据可视化（散点图）
    :param data: 数据集
    :return: 无返回
    """

    n = np.arange(len(x))

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()

def scatterPlot2(data):
    """
    数据可视化（散点图）
    :param data: 数据集
    :return: 无返回
    """
    x = data[:, 0]
    y = data[:, 1]
    n = np.arange(data.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    plt.show()



def plot_cluster_light(X, labels,list_light):
    """
    绘制带有标签的散点图。

    参数：
    X -- 二维数据集
    labels -- 数据点的标签

    返回：
    无
    """

    unique_labels = set(labels)
    colors = plt.cm.Spectral([i / float(len(unique_labels) - 1) for i in range(len(unique_labels))])

    for i, label in enumerate(unique_labels):
        x = [X[j][0] for j in range(len(X)) if labels[j] == label]
        y = [X[j][1] for j in range(len(X)) if labels[j] == label]
        plt.scatter(x, y, color=colors[i],alpha=0.5,s=100*(list_light[i]-min(list_light))/(max(list_light)-min(list_light)))

    for i, x in enumerate(X):
        plt.annotate(str(i), (x[0], x[1]))

    plt.show()


def plot_cluster(X, labels):
    """
    绘制带有标签的散点图。

    参数：
    X -- 二维数据集
    labels -- 数据点的标签

    返回：
    无
    """
    unique_labels = set(labels)
    colors = plt.cm.Spectral([i / float(len(unique_labels) - 1) for i in range(len(unique_labels))])

    for i, label in enumerate(unique_labels):
        x = [X[j][0] for j in range(len(X)) if labels[j] == label]
        y = [X[j][1] for j in range(len(X)) if labels[j] == label]
        plt.scatter(x, y, color=colors[i])

    for i, x in enumerate(X):
        plt.annotate(str(i), (x[0], x[1]))

    plt.show()
