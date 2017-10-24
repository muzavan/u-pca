# In this file, we will show how PCA in action on Digit Recognition case
# PCA will be utilized as noise filtering by simple idea: any components with variance much larget than the effect of the noise should be relatively unaffected by the noise

# Import Region
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from util import plot_digits

# Global Constant
_RANDOM_SEED = 42

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
plot_digits(digits.data)

# Now lets add some random noise to create a noisy dataset, and replot it\
np.random.seed(_RANDOM_SEED)
noisy = np.random.normal(digits.data, 4)
plot_digits(noisy)

# It's clear by eye that the images are noisy
# Let's train PCA on the noisy data, requesting the projection preserve 50% of the variance

pca = PCA(0.5).fit(noisy)
print(pca.n_components_)

# Then, we use this n_components to project digits to PC, then inverse it to see whether we have removed the noise
components = pca.transform(noisy)
filtered = pca.inverse_transform(components)
plot_digits(filtered)

 