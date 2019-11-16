import requests
import os
import pandas as pd
import umap
import numpy as np
#from ggplot import *
from dfply import *

import dfply as dplyr
from matplotlib import *
import seaborn as sns
#for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for PCA
from sklearn.decomposition import PCA
#for plotting
import matplotlib.pyplot as plt
# for dbscan
from sklearn.cluster import DBSCAN

import sklearn.cluster as cluster

from scipy import sparse





fly_data = pd.read_csv("MEMOIR/full_matrix_11132019_filterRows.csv")

# Fig 2
# Cluster all cells even if they dont have a barcode
fly_filtered = fly_data >>  mask(X.cell_size>1) >> mask(X.rowsum>0)

fly_filtered.shape


#where are the genes
cols = fly_filtered.columns
# old matrix : features = cols[2:10]
features = cols[5:13]

features


fly_filtered = fly_filtered.sample(4200)

x = fly_filtered.loc[:,features].div(fly_filtered.iloc[:,4], axis=0)
x

#x = x.loc[1:4000,:]

x.shape

x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components =6)
lowd_x = pca.fit_transform(x_scaled)

pca.explained_variance_ratio_.cumsum()


# WE can convert to sparse matrix such that UMAP does not take too much memory
sparse_lowd_x = sparse.csr_matrix(lowd_x)
sparse_lowd_x
# UMAP 1 for visualization
standard_embedding = umap.UMAP(random_state = 40).fit_transform(sparse_lowd_x)


# UMAP 2 this is a UMAP with different parameters, not for visualization but for clustering
clusterable_embedding = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=3,
    random_state=42,
).fit_transform(sparse_lowd_x)

clusterable_embedding

# # # CLUSTERING
dbscan_labels = DBSCAN(eps = 0.3).fit_predict(clusterable_embedding)

pd.value_counts(dbscan_labels)

clustered =(dbscan_labels >=0)



point_size = 1
f = plt.figure()
plt.scatter(standard_embedding[~clustered,0],standard_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=dbscan_labels[clustered],
            s=point_size, label = dbscan_labels[clustered],
            cmap='Spectral');
plt.ylabel("UMAP 1")
plt.xlabel("UMAP 2")
plt.show()


#if we are happy with the results we can save to data frame






#save the cluster labels
fly_filtered.loc[:,'clust_label'] = dbscan_labels
fly_filtered.loc[:,'umap_1'] = standard_embedding[:,0]
fly_filtered.loc[:,'umap_2'] = standard_embedding[:,1]



# https://seaborn.pydata.org/generated/seaborn.scatterplot.html

# https://seaborn.pydata.org/tutorial/color_palettes.html

# https://github.com/kieferk/dfply#the--and--pipe-operators

fly_clustered = fly_filtered >> mask(X.clust_label >=0)

a4_dims = (12, 12)

fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.scatterplot( ax = ax, x = 'umap_1',y = 'umap_2',hue = 'clust_label',data = fly_clustered,legend = 'full',palette = sns.color_palette("husl", len(pd.value_counts(dbscan_labels))-1))


fig.savefig("UMAP_noLog_JupyterLab_seaborn_ALLCELLS.pdf", bbox_inches='tight')


# plot gene clusters in the x,y space
in_situ_embedding = fly_filtered.loc[:,['xpos','ypos']].values

#choose which clusters to plot
clustered =(dbscan_labels >=0) & (dbscan_labels !=1)



fig = plt.figure()

a4_dims = (13, 13)


#fly_clustered_plot = fly_filtered >> mask(X.clust_label >=0) >> mask(X.clust_label !=1)

fly_clustered_plot

fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.scatterplot( ax = ax, x = 'xpos',y = 'ypos',hue = 'clust_label',data = fly_clustered,legend = 'full',palette = sns.color_palette("hls", len(pd.value_counts(dbscan_labels))-1))











##
