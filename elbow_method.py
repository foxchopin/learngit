#-*-coding: utf-8 -*-
"""
肘部法寻找最佳K值
从图中可以看出k<4时下降速度较快。
k=4时下降速度趋于平缓，所以k=4是最佳k值。
"""
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import numpy
data_path = '/Users/Xiaobang/jupyter/didi/'
train_data = pd.read_csv(data_path+'train_data_for_check_simple2.csv')
link_id = train_data['link_id']
del train_data['link_id']
train_data = train_data.values
length = train_data.shape[0]
clusters = [2,3,4,5,6,7,8]
K = range(1,10)
meandis = []
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(train_data)
    meandis.append(sum(np.min(cdist(train_data,kmeans.cluster_centers_,'euclidean'),axis=1))/length)
plt.plot(K,meandis)
    
    
    
    
