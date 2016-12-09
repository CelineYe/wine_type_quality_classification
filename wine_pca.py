import pandas as pd 
import matplotlib.pyplot as plt


#read in red wine dataset and white wine dataset
r_wine = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_red.csv', sep=(';'))
w_wine = pd.read_csv('E:/celine_sju/cus615/WineQualityData/winequality_white.csv', sep=(';'))

Xw_cols = w_wine.columns[0:11]
Yw_cols = w_wine.columns[11] 
Xw = w_wine[Xw_cols]
Yw = w_wine[Yw_cols]

Xr_cols = r_wine.columns[0:11]
Yr_cols = r_wine.columns[11]
Xr = r_wine[Xr_cols]
Yr = r_wine[Yr_cols]

# Normorlization data using Min-Max
from sklearn import preprocessing
minmax_scale1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Xw)
Xw_minmax = minmax_scale1.transform(Xw)
minmax_scale2 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(Xr)
Xr_minmax = minmax_scale2.transform(Xr)


# PCA for white wine
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
pca_transf_w = pca.fit_transform(Xw_minmax)
importance_w=pca.explained_variance_ratio_
importance_w, Xw_cols = zip(*sorted(zip(importance_w, Xw_cols)))

fig = plt.figure(figsize=(6, 4), dpi=80).add_subplot(111)
plt.bar(range(len(Xw_cols)), importance_w, align='center')
plt.xticks(range(len(Xw_cols)), Xw_cols, rotation='vertical')

plt.xlabel('Features')
plt.ylabel('Importance of features')
plt.title("PCA for white wine")
plt.show()

# PCA for red wine
pca_transf_r = pca.fit_transform(Xr_minmax)
importance_r=pca.explained_variance_ratio_
print importance_r
importance_r, Xr_cols = zip(*sorted(zip(importance_r, Xr_cols)))

fig = plt.figure(figsize=(6, 4), dpi=80).add_subplot(111)
plt.bar(range(len(Xr_cols)), importance_r, color='red', align='center')
plt.xticks(range(len(Xr_cols)), Xr_cols, rotation='vertical')

plt.xlabel('Features')
plt.ylabel('Importance of features')
plt.title("PCA for red wine")
plt.show()


# LDA for white wine
from sklearn.lda import LDA
lda = LDA(n_components=None)
transf_lda = lda.fit_transform(Xw_minmax, Yw)