import numpy as np
from bayesian_optimization_util import plot_convergence
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
from scipy.optimize import minimize



# Function 1
def f1(X, noise=0.2):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)


# Function 2
def f2(X, noise=0.2):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)


# Sum function (with a certain alpha)
def f_alpha(X, noise=0.2, alpha = 0.1):
    return f1(X,noise) + alpha * f2(X,noise)


f_list = [f1, f2]


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


if __name__ == '__main__':

    n_iter = 10
    bounds = np.array([[-1.0, 2.0]])
    noise = 0.2
    alpha = 0.5

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    # Noise-free objective function values at X
    Y1 = f1(X, 0)
    Y2 = f2(X, 0)
    Y_alpha = f_alpha(X, noise=0.0, alpha=alpha)

    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

    # Number of iterations
    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    # Initialize samples
    X_sample = np.array([[-0.9], [1.1]])
    Y1_sample = f1(X_sample,noise=noise)
    Y2_sample = f2(X_sample,noise=noise)
    Y_sample_list = [Y1_sample, Y2_sample]

    Y_sample_alpha = Y_sample_list[0] + alpha * Y_sample_list[1]
    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample_alpha)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample_alpha, gpr, bounds)

        # Obtain next noisy sample from the objective function
        Y_next_list = []
        for fun_idx, f in enumerate(f_list):
            Y_next_list.append(f(X_next, noise=noise))

        # Plot samples, surrogate function, noise-free objective and next sampling location
        plt.subplot(n_iter, 2, 2 * i + 1)
        plot_approximation(gpr, X, Y_alpha, X_sample, Y_sample_alpha, X_next, show_legend=i == 0)
        plt.title(f'Iteration {i + 1}')

        plt.subplot(n_iter, 2, 2 * i + 2)
        plot_acquisition(X, expected_improvement(X, X_sample, Y_sample_alpha, gpr), X_next, show_legend=i == 0)

        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        for fun_idx, Y_sample in enumerate(Y_sample_list):
            Y_sample_list[fun_idx] = np.vstack([Y_sample_list[fun_idx], Y_next_list[fun_idx]])

        Y_sample_alpha = Y_sample_list[0] + alpha * Y_sample_list[1]

    plot_convergence(X_sample, Y_sample_alpha)
