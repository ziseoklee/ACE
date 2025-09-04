from .distribution_distances import eot
from .mmd import linear_mmd2, mix_rbf_mmd2, poly_mmd2
from .optimal_transport import wasserstein
import ot as pot
import numpy as np

def compute_sample_based_metrics(a, b):
    w1 = wasserstein(a.double(), b.double(), power=1)
    w2 = wasserstein(a.double(), b.double(), power=2)
    mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=10 ** np.linspace(-2, 0, 10)).item()

    H_b, x_b, y_b = np.histogram2d(b[:, 0].cpu().numpy(), b[:, 1].cpu().numpy(), bins=200)
    H_a, x_a, y_a = np.histogram2d(a[:, 0].cpu().numpy(), a[:, 1].cpu().numpy(), bins=(x_b, y_b))
    total_var = 0.5 * np.abs(H_a / H_a.sum() - H_b / H_b.sum()).sum()

    return w1, w2, mmd_rbf, total_var
    
def compute_sample_based_metrics_1d(a, b):
    w1 = wasserstein(a.double(), b.double(), power=1)
    w2 = wasserstein(a.double(), b.double(), power=2)
    mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=10 ** np.linspace(-2, 0, 10)).item()

    return w1, w2, mmd_rbf
    