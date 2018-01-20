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
new_df.to_csv(data_path + 'new_df.csv',index=False,header=False)
fig = plt.figure(figsize=(20,20))
ax = Axes3D(fig,elev=-30,azim=0)

colors = ['black', 'blue', 'purple', 'yellow', 'red','green','cyan']
for i in range(6):
    sub_new_df = new_df[new_df['label']==i]
    plt.scatter(sub_new_df.pca_f1,sub_new_df.pca_f2,sub_new_df.pca_f3,c=colors[i])
