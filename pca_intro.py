# Import Region
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from util import draw_vector

# Global Constant
_RANDOM_STATE = 42

# This code will guide you to understand more about PCA
# The code used mainly taken or modified from : https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 

# First we will generate random point on X-Y axes
rng = np.random.RandomState(_RANDOM_STATE)
X = np.dot(rng.rand(2,2), rng.random(2,200)).T
plt.scatter(X[:,0], X[X,1])
plt.axis('equal')
plt.show()

# Based on our observation from the graph, it seems like that there are relation between x and y value
# If in regression task, we try to predict how to calculate y, in PCA we attempt to find relationship between x and y values
# This relation will be determined by finding a list of the principal axes in the data, and use those axes to describe the dataset

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X)

print(pca.components_) # Will print list of vector that can be new axes of our data
print(pca.explained_variance_) # Will print the variance score for each axes

# Plot data
plt.scatter(X[:,0], X[:,1], alpha=0.4)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_,pca.mean_ + v)

plt.axis('equal')
plt.show()

# On plt above, we show that we able to create a new axes based on data distribution (variance)
# Transformation from data axes to principal axes is composed of a translation, rotation, and uniform scaling.
# We can determine the order of principal axes by calculating its variance score (the more variance the early)
# The first, second, and so on principal axes should not have any covariance relation

# PCA as a dimensionality reduction
# We will try to see the effect if we only take the first principal component
pca = PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
print("Original shape: ", X.shape)
print("Transformed shape: ", X_pca.shape)

# To show/prove whether using 1 principal component (PC) can represent the data, we will inverse transform it
X_new = pca.inverse_transform(X_pca)
plt.scatter(X[:,0],X[:,1],alpha=0.4) # Original data
plt.scatter(X_new[:,0], X_new[:,1], alpha=0.8) # The inversed data (converted to PC1 only (which means, the value of PC2 is 0) back to "original" value)
plt.axis('equal')
plt.show()

# You should see that the interted data looks like the regression line, because we disable the variance in the PC2 axis
# In this example, we still be able to see the distribution of data, even when we ignore the PC2 value
# This is where PCA comes in handy, we will be able to reduce our dimension by setting up some threshold in which we can safely ignore the loss information
