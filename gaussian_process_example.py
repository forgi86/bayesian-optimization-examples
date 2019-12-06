import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

# Objective function
noise = 0.2


def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

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

    bounds = np.array([[-1.0, 2.0]])
    X_init = np.array([[-0.9], [1.1]])
    Y_init = f(X_init)

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

    # Noise-free objective function values at X
    Y = f(X, 0)

    # Plot optimization objective with noise level
    plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
    plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
    plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
    plt.legend()




    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

    # Initialize samples
    X_sample = np.copy(X_init)
    Y_sample = np.copy(Y_init)


    # Number of iterations
    n_iter = 10

    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    for i in range(n_iter):
        # Update Gaussian process with existing samples
        gpr.fit(X_sample, Y_sample)

        # Obtain next sampling point from the acquisition function (expected_improvement)
        X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

        # Obtain next noisy sample from the objective function
        Y_next = f(X_next, noise)

        # Plot samples, surrogate function, noise-free objective and next sampling location
        #plt.subplot(n_iter, 2, 2 * i + 1)
        #plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i == 0)
        #plt.title(f'Iteration {i + 1}')
        # plt.grid(True)

        #plt.subplot(n_iter, 2, 2 * i + 2)
        #plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i == 0)
        # plt.grid(True)

        # Manually plot #
        plt.figure()
        mu, std = gpr.predict(X, return_std=True)
        plt.fill_between(X.ravel(),
                         mu.ravel() + 1.96 * std,
                         mu.ravel() - 1.96 * std,
                         alpha=0.1)
        plt.plot(X, Y, 'y--', lw=1, label=r'$J(\theta)$')
        plt.xlabel(r"Design parameter $\theta$")
        plt.xlabel(r"Performance index $J(\theta)$")
        plt.grid()

        plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
        plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
        plt.legend()


        # Add sample to previous samples
        X_sample = np.vstack((X_sample, X_next))
        Y_sample = np.vstack((Y_sample, Y_next))
