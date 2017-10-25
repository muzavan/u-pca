# In this file, we will show how PCA in action on Digit Recognition case
# PCA will be utilized as noise filtering by simple idea: any components with variance much larget than the effect of the noise should be relatively unaffected by the noise

# Import Region
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from util import plot_digits
import numpy as np

# Global Constant
_RANDOM_SEED = 42

# Load the data from sklearn datasets
digits = load_digits()

# Let's first show the actual image from dataset
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

 