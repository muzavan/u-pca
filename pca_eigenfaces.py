# In this example, we will try to mimick the base face of our data set

# Import Region
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.datasets import fetch_lfw_people
from matplotlib import pyplot as plt

# Global Constants
_RANDOM_SEED = 42

# Get the data set
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

# Because this is large dataset, we will try to use RandomizedPCA -- it contains a randomized method to approximate the first N principal components much more quickly than the standard PCA estimator.
pca = RandomizedPCA(150) # We try to fetch first 150 components
pca.fit(faces.data)

# Then, we try to show the image represented by several components
fig, axes = plt.subplots(3,8,figsize=(9,4), subplot_kw={'xticks' : [], 'yticks' : []}, gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(faces.images.shape[1], faces.images.shape[2]), cmap='bone')
plt.show()

# In image above, you should could check, that at first the principal components are the base face structure, then we moved to recognize face features like nose, eyes, mouth, etc

# Let's find out the cumulative some of the variance to determine how much PCs suitable for our case
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components: ')
plt.ylabel('Cumulative explained variance: ')
plt.show()

# From the plot above, we can safely assume that using 150 PCs alreaed retrieve ~90% of our variances
# Let's see it by comparing the original image with image using only 150PCs
pca = RandomizedPCA(150).fit(faces.data)
components = pca.transform(faces.data)
pca_faces = pca.inverse_transform(components)

# Plot the results
fig, ax = plt.subplots(2,10,figsize=(10,2.5),subplot_kw={'xticks' : [], 'yticks' : []}, gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i in range(10):
    dimensionH = faces.images.shape[1]
    dimensionW = faces.images.shape[2]
    ax[0,i].imshow(faces.data[i].reshape(dimensionH, dimensionW), cmap='binary_r')
    ax[1,i].imshow(pca_faces[i].reshape(dimensionH, dimensionW), cmap='binary_r')

plt.show()