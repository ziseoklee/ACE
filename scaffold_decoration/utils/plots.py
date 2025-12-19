import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(values, save_path, bins=10, xlabel="Score", ylabel="Frequency", title="Distribution of Scores"):
    """
    Plot and save a histogram of values.
    
    Args:
        values: Array-like data to plot
        save_path: Path to save the plot
        bins: Number of histogram bins (default: 10)
        xlabel: Label for x-axis (default: "Score") 
        ylabel: Label for y-axis (default: "Frequency")
        title: Plot title (default: "Distribution of Scores")
    """
    plt.figure()
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

# If you want a smooth distribution (KDE-like), you can overlay a density plot
from scipy.stats import gaussian_kde

def plot_kde_histogram(values, save_path, bins=10, xlabel="Score", ylabel="Density", 
                      title="Distribution of Scores with KDE", bandwidth=None):
    """
    Plot and save a histogram with Kernel Density Estimation (KDE) overlay.
    
    Args:
        values: Array-like data to plot
        save_path: Path to save the plot
        bins: Number of histogram bins (default: 10)
        xlabel: Label for x-axis (default: "Score")
        ylabel: Label for y-axis (default: "Density") 
        title: Plot title (default: "Distribution of Scores with KDE")
        bandwidth: KDE bandwidth parameter (default: None, uses Scott's rule)
    """
    plt.figure()
    density = gaussian_kde(values, bw_method=bandwidth)
    xs = np.linspace(values.min() - 1, values.max() + 1, 200)
    plt.hist(values, bins=bins, density=True, alpha=0.5, edgecolor="black")
    plt.plot(xs, density(xs), 'r-', label="KDE")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()