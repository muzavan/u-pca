# In this file, we will show how PCA in action on Digit Recognition case

# Import Region
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Load the data from sklearn datasets
digits = load_digits()
print(digits.data.shape) # (1797,64) : 1797 sampe with 8x8 image

# We will try to project this data to 2 dimension (so can visualized easily :p)
pca = PCA(n_components=2)
projected = pca.fit_transform(digits.data)
print("{begin} transformed to {end}".format(begin=digits.data.shape, end=projected.shape))

# Now, we try to plot each data to PC1 and PC2 axes, and see where we can separate the data linearly
plt.scatter(projected[:,0],projected[:,1], c = digits.target, edgecolor="none", alpha=0.5, cmap=plt.cm.get_cmap('spectral',10))
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

# On the image above, we can see that in 2 PCs, we (more or less) already can grouped each digits in a certain region

# Why we have to create PC instead of using the original value?
# In digit recognition, using n-th pixel of the canvas won't retrieve much information, while in PCA it will give more information (becasue each PC is combined features)

# Choosing the number of components
# This cam be determined by looking at the cumulative explained variance ratio as a function of the number of components
pca = PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ration_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.show()

# On the image above, we can see that we can get 75% of the variance by just using 10 components. Please also note that we can close up to 100% just by using 50 PCs.
