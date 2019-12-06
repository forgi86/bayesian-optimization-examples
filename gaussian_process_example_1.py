import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import os

# Objective function
noise = 0.01


def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)


if __name__ == '__main__':

    np.random.seed(1)

    n_iter = 20
    bounds = np.array([[-1.0, 2.0]])

    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(5.0) * Matern(length_scale=0.1, nu=10.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

    X_init = np.array([[]]).T
    Y_init = np.array([[]]).T


    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    # Noise-free objective function values at X
    Y = f(X, 0)

    # Initialize samples
    X_sample = np.copy(X_init)
    Y_sample = np.copy(Y_init)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        if i > 0:
            gpr.fit(X_sample, Y_sample)

        # Manually plot #
        plt.figure(figsize=(5, 4))
        mu, std = gpr.predict(X, return_std=True)
        plt.plot(X, mu, 'b-', lw=1, label='GP mean')
        plt.fill_between(X.ravel(),
                         mu.ravel() + 1.96 * std,
                         mu.ravel() - 1.96 * std,
                         alpha=0.1,
                         color='c')
        plt.plot(np.NaN, np.NaN, 'c', linewidth=4, alpha=0.1, label='GP 95% c.i.')
        plt.plot(X, Y, 'k', lw=1, label=r'$J(\theta)$')
        plt.xlabel(r"Design parameter $\theta$")
        plt.ylabel(r"Performance index $J(\theta)$")
        plt.ylim([-2.5, 2.5])
        plt.grid()
        if i > 0:
            plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fig_filename = f'GP_fit_{i}.pdf'
        plt.savefig(os.path.join('fig', fig_filename))


        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = np.random.rand(1)*3 - 1

        # Obtain next noisy sample from the objective function
        Y_next = f(X_next, noise)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
