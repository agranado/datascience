


import requests
import os
import pandas as pd
import umap
import numpy as np
from ggplot import *
from dfply import *

#for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for PCA
from sklearn.decomposition import PCA
# for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#for plotting
import matplotlib.pyplot as plt
# for dbscan
from sklearn.cluster import DBSCAN

import sklearn.cluster as cluster
#get the iris dataset
data = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

#open with 'w' will destroy the dataset if it already exists
with open('iris.dat','w') as f:
     f.write(data.text)


#real retina data is here:
# read the input file with pandas
df = pd.read_csv('retinaComposition.csv')

df = pd.read_csv('MEMOIR/gene_expression_labeled.txt')

# RETINA
#replace NaN values (empty cells) with 0
df = df.fillna(0)

df.head()
#    record #  retina C/P  age  cone  gang  horiz  ama  rods  bip  mul
# 0         1     305   c  e14   0.0   1.0    0.0  0.0   0.0  0.0  0.0
# 1         2     305   c  e14   0.0   0.0    0.0  1.0   0.0  0.0  0.0
# 2         3     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
# 3         4     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
# 4         5     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
features = ['cone', 'gang', 'horiz', 'ama','rods','bip','mul']


# # # # # # #
# For UMAP MEMOMIR
# # # # # # # # #
len(np.unique(df.loc[:,'Barcode']))
cols = df.columns
features = cols[2:]

# READ x,y position (in situ/ experimental)
centroids = pd.read_csv("MEMOIR/centroids.txt",header = None,names = ['x','y'])
#
in_situ_embedding = centroids.loc[:,['x','y']].values

# CLONES
clones = pd.read_csv("MEMOIR/clones.txt",header = None,names =['clone_id'])

# # # # # # # # # GENERAL

#We can now separate the features
x = df.loc[:,features].values



# CORE

# raw.data (7dim) -> scaled -> PCA (4dim?) -> UMAP-> DBSCAN -> UMAP

# So the results are still not great

# NEXT we will try to use UMAP to reduce dimension and create a clusterable_embedding
# see https://umap-learn.readthedocs.io/en/latest/clustering.html

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
dbscan_labels = DBSCAN(eps = 0.3).fit_predict(clusterable_embedding)

#How many clusters
np.unique(dbscan_labels)

# Size distribution of clusters
pd.value_counts(dbscan_labels)









# # # # # PLOT
# Plot the data in the clusterable embedding, here it should look good, maybe a bit spread
plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 1],c=dbscan_labels, s=point_size, cmap='Spectral');
plt.show()

#Check how it looks on standard embedding
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1],c=dbscan_labels, s=point_size, cmap='Spectral');
plt.show()



# IF THERE ARE MORE THAN 2 components in the clusterable umap:
plt.scatter(clusterable_embedding[:, 1], clusterable_embedding[:, 2],c=dbscan_labels, s=point_size, cmap='Spectral');
plt.show()

plt.scatter(clusterable_embedding[:, 0], clusterable_embedding[:, 2],c=dbscan_labels, s=point_size, cmap='Spectral');
plt.show()


# SAVE PLOT
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


#save to pdf
f.savefig("MEMOIR/UMAP_fly.pdf", bbox_inches='tight')



#colors=plt.cm.get_cmap('Spectral', 15)
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
f.savefig("MEMOIR/UMAP_inSitu.pdf", bbox_inches='tight')










########
# # # # #
# # # # # MAKE Plot using the clone_id (barcodes)

clones = pd.read_csv("MEMOIR/fly_data_clone_id.txt")

clones.head()

clone_labels = clones.loc[:,'clone_id'].values
clone_labeled = (clone_labels==16)

len(clone_labels)
len(clone_labeled)
sum(clone_labeled)
#colors=plt.cm.get_cmap('Spectral', 15)



point_size = 6
f = plt.figure()
plt.scatter(standard_embedding[~clone_labeled,0],standard_embedding[~clone_labeled,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(standard_embedding[clone_labeled, 0],
            standard_embedding[clone_labeled, 1],
            c=clone_labels[clone_labeled],
            s=point_size,
            cmap='Spectral');
plt.ylabel("UMAP 1")
plt.xlabel("UMAP 1")
plt.show()

f.savefig("MEMOIR/UMAP_byClones.pdf", bbox_inches='tight')










#Save all data into a new DataFrame
allData = pd.concat([ df , centroids , pd.DataFrame( standard_embedding ), pd.DataFrame(dbscan_labels) ], axis = 1)


allData.head()

allData.to_csv('FlyData_clusteredUMAP.txt',sep='\t',index = False)

allData.head()



fly_data = pd.read_csv("MEMOIR/fly_data_clone_id.txt")
fly_data.head()



pp = fly_data >> mask(X.clust_label >=0 , X.clone_id >0)


p = ggplot(aes(x = 'clone_id'),data = fly_data) + geom_bar()
p.show()

# fig, ax  =  plt.subplots()

p = ggplot(aes(x = 'fru', y = 'Gad1'),data = fly_data) + geom_point()
p.show()
# ax.scatter(standard_embedding[~clustered,0],standard_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)
#
#
# scatter = ax.scatter(standard_embedding[clustered, 0],
#             standard_embedding[clustered, 1],
#             c=dbscan_labels[clustered],
#             s=point_size,
#             cmap='plasma');
#
#
#
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
# ax.add_artist(legend1)
#
#
# plt.ylabel("UMAP 1")
# plt.xlabel("UMAP 1")
# plt.show()









# biplot
import numpy as np

def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    #plt.scatter(xs * scalex,ys * scaley, c = y,cmap = 'plasma')

    plt.scatter(xs[~clustered]*scalex,ys[~clustered]*scaley,c = (0.5,0.5,0.5),s = point_size,alpha =0.5)
    plt.scatter(xs[clustered]*scalex,ys[clustered]*scaley,c=y[clustered],s=point_size,cmap='plasma');

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')


#make plot
f = plt.figure()
plt.xlim(-0.25,0.75)
plt.ylim(-0.75,1)
plt.xlabel("PC{}".format(1))
plt.ylabel("PC{}".format(2))
plt.grid()

y = dbscan_labels
myplot(lowd_x[:,0:2] ,np.transpose(pca.components_[0:2,:]),labels = features)
plt.show()


f.savefig("biplot_retinaE14.pdf", bbox_inches='tight')

# # # # # #
 # # # # # # 
# # # # # #
# # # # # # 

# perform on randomized dataset

x_rand = x

for i in range(0,4):
    x_rand[:,i] = shuffle(x[:,i])



























# We can now scale the data  mean = 0, sd = 1

x_scaled = StandardScaler().fit_transform(x)
x_scaled

#let's perform PCA

pca = PCA(n_components =2)
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents
                , columns = ['pc1','pc2'])

principalDf



plt.scatter(principalDf["pc1"],principalDf["pc2"])
plt.show()



# from: https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea
dbscan = DBSCAN(eps = 0.5,min_samples = 5)
clusters  = dbscan.fit_predict(x_scaled)
#
# plt.scatter(x[:,0],x[:,1],c = clusters, cmap = "plasma")
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 0')
# plt.show()




plt.scatter(principalDf["pc1"],principalDf["pc2"], c = clusters, cmap = "plasma")
plt.show()


# # # # #
# UMAP
# we are going to perform UMAP reduction on the scaled dataset,
# we are going to color them by the clusters found by DBSCAN...
import umap
standard_embedding = umap.UMAP(random_state = 42).fit_transform(x_scaled)
plt.scatter(standard_embedding[:,0],standard_embedding[:,1],c=clusters,cmap='plasma')
plt.show()



# # We can also perform k-means on the UMAP reduced dataset
# # this is NOT GOOD !
# kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(x_scaled)s
# plt.scatter(standard_embedding[:,0],standard_embedding[:,1],c=kmeans_labels, s = 1, cmap = "plasma")
# plt.show()


# FROM https://umap-learn.readthedocs.io/en/latest/clustering.html
#DBSCAN on scaled Data
dbscan = DBSCAN(eps = 0.5,min_samples = 5)
clusters  = dbscan.fit_predict(x_scaled)




# DBSCAN on PCA (4 PCs)
# 1. Perform PCA on scaled data

x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components =3)
lowd_x = pca.fit_transform(x_scaled)

# 3. UMAP on scaled data set
# we are going to perform UMAP reduction on the scaled dataset,
# we are going to color them by the clusters found by DBSCAN...

standard_embedding = umap.UMAP(random_state = 42).fit_transform(lowd_x)
# plt.scatter(standard_embedding[:,0],standard_embedding[:,1],c=clusters,cmap='plasma')
# plt.show()



# 2. DBSCAN clustering on low dimensional space
dbscan_labels = DBSCAN(eps = 0.8,min_samples = 3).fit_predict(lowd_x)



# lets plot the data points that were NOT clustered by DBSCAN (cluster ID 0 ) as gray point
clustered =(dbscan_labels >=0)

point_size = 6
plt.scatter(standard_embedding[~clustered,0],standard_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=dbscan_labels[clustered],
            s=point_size,
            cmap='plasma');
plt.show()
