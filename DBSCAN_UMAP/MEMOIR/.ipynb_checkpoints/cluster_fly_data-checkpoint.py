# cluster fly data

import requests
import os
import pandas as pd
import umap
import numpy as np
#from ggplot import *
from dfply import *
#from plotnine import *

#for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for PCA
from sklearn.decomposition import PCA
#for plotting
import matplotlib.pyplot as plt
# for dbscan
from sklearn.cluster import DBSCAN

import sklearn.cluster as cluster


##

fly_data = pd.read_csv("MEMOIR/fly_data_clone_id.txt")
fly_data.head()

# let's filter for cells with actual clone ID

fly_filtered = fly_data >> mask(X.clust_label >=0 , X.clone_id >0)

fly_filtered.head()
fly_filtered.shape

#where are the genes
cols = fly_filtered.columns
features = cols[2:10]
features

#
x = fly_filtered.loc[:,features].values


x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components =6)
lowd_x = pca.fit_transform(x_scaled)

pca.explained_variance_ratio_.cumsum()

# UMAP 1 for visualization
standard_embedding = umap.UMAP(random_state = 42).fit_transform(lowd_x)


# UMAP 2 this is a UMAP with different parameters, not for visualization but for clustering
clusterable_embedding = umap.UMAP(
    n_neighbors=25,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(lowd_x)

clusterable_embedding

# # # CLUSTERING
dbscan_labels = DBSCAN(eps = 0.4).fit_predict(clusterable_embedding)

#How many clusters
np.unique(dbscan_labels)

# Size distribution of clusters
pd.value_counts(dbscan_labels)




# UMAP PLOT
# Fitler for cluster -1
clustered =(dbscan_labels >=0)


point_size = 6
f = plt.figure()
plt.scatter(standard_embedding[~clustered,0],standard_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=dbscan_labels[clustered],
            s=point_size,
            cmap='Spectral');
plt.ylabel("UMAP 1")
plt.xlabel("UMAP 1")
plt.show()




# plot gene clusters in the x,y space
in_situ_embedding = fly_filtered.loc[:,['x','y']].values

point_size = 4
f = plt.figure()
plt.scatter(in_situ_embedding[~clustered,0],in_situ_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(in_situ_embedding[clustered, 0],
            in_situ_embedding[clustered, 1],
            c=dbscan_labels[clustered]*4,
            s=point_size,
            cmap='Spectral');
plt.ylabel("in situ X")
plt.xlabel("in situ Y")
plt.show()



# PLOT clones into the UMAP space

range(20)

clone_labels = fly_filtered.loc[:,'clone_id'].values

fly_filtered.clone_id.value_counts()
#
# # SAVE plots one by oneq§
# for i in range(21):
#     clone_labeled = (clone_labels==i)
#
#     point_size = 6
#     f = plt.figure()
#     plt.scatter(standard_embedding[~clone_labeled,0],standard_embedding[~clone_labeled,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)
#
#
#     plt.scatter(standard_embedding[clone_labeled, 0],
#                 standard_embedding[clone_labeled, 1],
#                 c=clone_labels[clone_labeled],
#                 s=point_size,
#                 cmap='Spectral');
#     plt.ylabel("UMAP 1")
#     plt.xlabel("UMAP 1")
#     plt.title('Clone' + str(i) + ' mapped into the gene-expression space')
#     plt.show()
#
#     f.savefig("MEMOIR/clones/" + str(i) +  ".pdf", bbox_inches='tight')
#
#
#
# # # # Cluster composition
# ggplot(aes(x='clone_id',fill = 'clust_label'),data = fly_filtered) + geom_bar(stat='count')
fly_filtered.head()
fly_filtered.to_csv('MEMOIR/flyData_filtered_clustered.txt',header = False,index = False)


#fly_data.to_csv('MEMOIR/flyData_filtered_clustered.txt')
