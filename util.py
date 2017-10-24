# Import Region
from matplotlib import pyplot as plt

def draw_vector(v0, v1, ax=None):
    """
    Draw arrow-style vector based on v0 and v1
    """
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle='-->', linewidth=2, shrinkA=0, shrinkB=0)
    ax.annonate('', v1, v0, arrowprops=arrowprops)
    plt.show()

def plot_digits(data):
    """
    Show the image of digits as a graph plot
    """
    fig, axes = plt.subplots(4, 10, figsize=(10,4), subplot_kw={'xticks' : [], 'yticks' : []},gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap='binary', interpolation='nearest', clim=(0,16))

    plt.show()


