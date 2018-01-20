import pandas as pd
data_path = '/Users/Xiaobang/jupyter/didi/'
data = pd.read_csv(data_path + 'result_check.csv')
data.head(5)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
train_data = data.copy()
label = train_data.label
train_data = train_data.drop(['link_id','label'],axis=1)
pca = PCA(n_components = 3)
pca.fit(train_data)
new_data = pca.transform(train_data)

new_df = pd.DataFrame(new_data)
new_df = pd.concat([new_df, label],axis=1)
new_df.columns = ['pca_f1','pca_f2','pca_f3','label']
fig = plt.figure(figsize=(20,20))
ax = Axes3D(fig)
plt.scatter(new_data[:,0],new_data[:,1],new_data[:,2])
