import pandas as pd
import umap
import numpy as np

#for scaling the dataset
from sklearn.preprocessing import StandardScaler
# for PCA
from sklearn.decomposition import PCA
#for plotting
import matplotlib.pyplot as plt
# for dbscan
from sklearn.cluster import DBSCAN
# general clustering functions
import sklearn.cluster as cluster


#Graphic parameters for exporting
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

#real retina data is here:
# read the input file with pandas
df = pd.read_csv('retinaComposition.csv')
#replace NaN values (empty cells) with 0
df = df.fillna(0)

df.head()
#    record #  retina C/P  age  cone  gang  horiz  ama  rods  bip  mul
# 0         1     305   c  e14   0.0   1.0    0.0  0.0   0.0  0.0  0.0
# 1         2     305   c  e14   0.0   0.0    0.0  1.0   0.0  0.0  0.0
# 2         3     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
# 3         4     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
# 4         5     305   c  e14   1.0   0.0    0.0  0.0   0.0  0.0  0.0
features = ['Cone', 'Gang', 'Horiz', 'Ama','Rods','Bip','Mul']

#We can now separate the features
x = df.loc[:,features].values





#

# 1. Scaling data and PCA
x_scaled = StandardScaler().fit_transform(x)
pca = PCA(n_components =4)
lowd_x = pca.fit_transform(x_scaled)


# 2. NEXT we will try to use UMAP to reduce dimension and create a clusterable_embedding
# see https://umap-learn.readthedocs.io/en/latest/clustering.html
# this is a UMAP with different parameters, not for visualization but for clustering
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(lowd_x)

# 3. Cluster the UMAP embedding using DBSCAN
# see https://towardsdatascience.com/dbscan-clustering-for-data-shapes-k-means-cant-handle-well-in-python-6be89af4e6ea
dbscan_labels = DBSCAN().fit_predict(clusterable_embedding)

# DBSCAN returns -1 for datapoints that was not able to cluster
clustered =(dbscan_labels >=0)

# 4. MAkE plot using standard_embedding (visualization UMAP, more compact )

# UMAP (for visualization)
standard_embedding = umap.UMAP(random_state = 42).fit_transform(lowd_x)


point_size = 6
f = plt.figure()
plt.scatter(standard_embedding[~clustered,0],standard_embedding[~clustered,1],c = (0.5,0.5,0.5),s = point_size,alpha =0.5)


plt.scatter(standard_embedding[clustered, 0],
            standard_embedding[clustered, 1],
            c=dbscan_labels[clustered],
            s=point_size,
            cmap='plasma');
plt.ylabel("UMAP 1")
plt.xlabel("UMAP 1")
plt.show()



#SAVE  Figure 1 UMAP

f.savefig("UMAP_retinaE14_v2.pdf",  bbox_inches='tight',transparent=True)



# # # # # BIPLOT from PCA


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

f.savefig("biplot_retinaE14_v2.pdf",  bbox_inches='tight',transparent=True)
