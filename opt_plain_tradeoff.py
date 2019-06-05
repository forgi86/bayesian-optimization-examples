import numpy as np
from bayesian_optimization_util import plot_convergence
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def f_vec(X, noise=0.1):
    y1 = X ** 2 + noise * np.random.randn(*X.shape)
    y2 = 4 * X + 1 + noise * np.random.randn(*X.shape)
    y_vec = np.hstack([y1, y2])
    return y_vec


def f_tradeoff(y_vec, alpha):
    alpha_vec = np.hstack((1, alpha))
    y_tradeoff = y_vec @ alpha_vec
    y_tradeoff = y_tradeoff.reshape(-1,1)
    return y_tradeoff


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
    ALPHA_VEC = np.arange(0, 1.1, 0.1)
    n_alpha = len(ALPHA_VEC)
    bounds = np.array([[-3.0, 3.0]])

    # Gaussian process with Matérn kernel as surrogate model
    m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
    gpr = GaussianProcessRegressor(kernel=m52, alpha = 0.1**2)

    X_init = np.array([[-2.],
                       [-1.],
                       [0.],
                       [1.],
                       [2.]
                      ]
                     )
    n_init = np.shape(X_init)[0]
    Y_init = f_vec(X_init)

    X_sample = np.copy(X_init)
    J_sample = np.copy(Y_init)
    for iter_idx in range(n_iter):
        for alpha_idx in range(len(ALPHA_VEC)):

            # Consider problem with current alpha
            alpha = ALPHA_VEC[alpha_idx]
            J_sample_alpha = f_tradeoff(J_sample, alpha)

            J_sample_alpha = - J_sample_alpha # in case of minimization
            gpr.fit(X_sample, J_sample_alpha)   # fit GP for current sample

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = propose_location(expected_improvement, X_sample, J_sample_alpha, gpr, bounds)

            # Compute vector output for next sample
            J_next = f_vec(X_next)

            X_sample = np.vstack([X_sample,
                                  X_next])
            J_sample = np.vstack([J_sample,
                                  J_next])


    N_test = np.shape(X_sample)[0]
    J_tradeoff_all = np.zeros((N_test, n_alpha))

    for alpha_idx in range(n_alpha):
        alpha = ALPHA_VEC[alpha_idx]
        J_tradeoff_all[:, alpha_idx] = f_tradeoff(J_sample, alpha).ravel()

    ix_opt_alpha = np.argmin(J_tradeoff_all, axis=0)
    x_opt_BO = X_sample[ix_opt_alpha,:]
    y_opt_BO = np.diag(J_tradeoff_all[ix_opt_alpha, :])

    b = 4*ALPHA_VEC
    a = 1
    c = 1
    delta = b**2 - 4*ALPHA_VEC*c
    x_opt_true = (-2*ALPHA_VEC)/2
    y_opt_true = -delta/(4*a)

    plt.figure()
    plt.plot(x_opt_true, x_opt_BO, '*')
    plt.xlabel('$x_{opt}(\\alpha)$')
    plt.ylabel('$\\hat x_{opt}(\\alpha)$')
    plt.grid(True)

    plt.figure()
    plt.plot(y_opt_true, y_opt_BO, '*')
    plt.xlabel('$J(x_{opt}(\\alpha))$')
    plt.ylabel('$J(\\hat x_{opt}(\\alpha))$')
    plt.grid(True)

    # Recalc on a dense lattice for 3d plots
    X_VEC_DENSE = np.arange(-3, 3, 0.01).reshape(-1, 1)
    ALPHA_VEC_DENSE = np.arange(-0.1, 1.1, 0.01)

    b = 4 * ALPHA_VEC_DENSE
    a = 1
    c = 1
    delta = b ** 2 - 4 * ALPHA_VEC_DENSE * c

    X_MIN_DENSE = (-b) / 2
    Y_MIN_DENSE = -delta / (4 * a)

    Xx, Aa = np.meshgrid(X_VEC_DENSE, ALPHA_VEC_DENSE)
    Jj = np.zeros(Xx.shape)
    for idx_alpha in range(len(ALPHA_VEC_DENSE)):
        alpha = ALPHA_VEC_DENSE[idx_alpha]
        X_vec = Xx[idx_alpha, :].reshape(-1,1)
        Y_vec = f_vec(X_vec, noise=0.0)
        J_sample_alpha = f_tradeoff(Y_vec, alpha)
        Jj[idx_alpha, :] = J_sample_alpha.ravel()

    plt.figure()
    plt.contour(Xx, Aa, Jj)
    plt.plot(X_MIN_DENSE, ALPHA_VEC_DENSE, 'k--')

    X_sample_alpha = X_sample[n_init:, :].reshape(-1, n_alpha)
    for alpha_idx in range(n_alpha):
        alpha_val =  ALPHA_VEC[alpha_idx]
        x_plot = X_sample_alpha[:, alpha_idx].ravel()
        alpha_plot = alpha_val*np.ones(np.shape(x_plot))
        plt.plot(x_plot, alpha_plot, '*')

    plt.plot(x_opt_BO, ALPHA_VEC, 'or')

    X_sample_init = X_sample[:n_init, :]
    alpha_val = 0
    for init_idx  in range(n_init):
        x_plot = X_sample_init[init_idx,:].ravel()
        alpha_plot = alpha_val * np.ones(np.shape(x_plot))
        plt.plot(x_plot, alpha_plot, 'sy')

    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('$\\alpha$')

