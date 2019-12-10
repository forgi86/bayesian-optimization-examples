import numpy as np
from bayesian_optimization_util import plot_convergence
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from objective import  ObjectiveTradeOff


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

    f_obj = ObjectiveTradeOff(nx=1,ny=2,alpha=0.1) # alpha value arbitrary here

    n_iter = 20
    ALPHA_VEC = np.arange(0, 1, 0.1)
    n_alpha = len(ALPHA_VEC)

    bounds = np.array([[-2.0, 2.0]])

    # Dense grid of points within bounds
    X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
    Y = f_obj.f_x(X, 0)

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
    Y_init = f_obj.f_x(X_init)

    for iter_idx in range(n_iter):
        for alpha_idx in range(len(ALPHA_VEC)):

            # Consider problem with current alpha
            alpha = ALPHA_VEC[alpha_idx]

            # Not really used
            f_obj.set_alpha(alpha)

            X_init = f_obj.X_data
            Y_init = f_obj.Y_data[:, 0] + alpha * f_obj.Y_data[:, 1]
            Y_init = Y_init.reshape(-1, 1) # to column vector
            Y_init = -Y_init # API was meant for maximization!
            # Update Gaussian process with existing samples
            gpr.fit(X_init, Y_init)

            # Obtain next sampling point from the acquisition function (expected_improvement)
            X_next = propose_location(expected_improvement, X_init, Y_init, gpr, bounds)

            Y_next = f_x(X_next, store_objective=True) # do not need the value, just need it to be stored

    N_test = np.shape(f_obj.X_data)[0]
    Y_tradeoff = np.zeros((N_test, n_alpha))
    X_data = np.copy(f_obj.X_data)
    Y_data = np.copy(f_obj.Y_data)

    for alpha_idx in range(n_alpha):
        alpha = ALPHA_VEC[alpha_idx]
        Y_tradeoff[:, alpha_idx] = Y_data[:,0] + alpha * Y_data[:,1]

    # J = x**2 + 2*alpha*x + 1

    ix_opt_alpha = np.argmin(Y_tradeoff, axis=0)

    b = 4*ALPHA_VEC
    a = 1
    c = 1
    delta = b**2 - 4*ALPHA_VEC*c

    xmin_true = (-2*ALPHA_VEC)/2
    ymin_true = -delta/(4*a)

    xmin_BO = X_data[ix_opt_alpha,:]
    ymin_BO = np.diag(Y_tradeoff[ix_opt_alpha, :])

    plt.figure()
    plt.plot(xmin_true, xmin_BO, '*')
    plt.xlabel('$x_{opt}(\\alpha)$')
    plt.ylabel('$\\hat x_{opt}(\\alpha)$')
    plt.grid(True)

    plt.figure()
    plt.plot(ymin_true, ymin_BO, '*')
    plt.xlabel('$J(x_{opt}(\\alpha))$')
    plt.ylabel('$J(\\hat x_{opt}(\\alpha))$')
    plt.grid(True)

    # In[3d plot]

    X_VEC_DENSE = np.arange(-2,2, 0.01).reshape(-1, 1)
    ALPHA_VEC_DENSE = np.arange(-0.1, 1.1, 0.01)

    b = 4*ALPHA_VEC_DENSE
    a = 1
    c = 1
    delta = b**2 - 4*ALPHA_VEC_DENSE*c

    X_MIN_DENSE = (-b)/2
    Y_MIN_DENSE = -delta/(4*a)



    Xx, Aa = np.meshgrid(X_VEC_DENSE, ALPHA_VEC_DENSE)
    Jj = np.zeros(Xx.shape)
    for idx_alpha in range(len(ALPHA_VEC_DENSE)):
        alpha = ALPHA_VEC_DENSE[idx_alpha]
        f_obj.set_alpha(alpha)
        f_obj.noise = 0.0
        Jj[idx_alpha,:] = f_obj.f_x(Xx[idx_alpha,:].reshape(-1,1), store_objective=False).ravel()
                
    plt.figure()
    plt.contour(Xx, Aa, Jj);
    plt.plot(X_MIN_DENSE, ALPHA_VEC_DENSE, 'k--')
    plt.plot(xmin_BO, ALPHA_VEC, '*r')
    plt.grid(True)

    X_data_alpha = X_data[n_init:,:].reshape(-1,n_alpha)
    for alpha_idx in range(n_alpha):
        plt.plot(X_data_alpha[:,alpha_idx].ravel(), ALPHA_VEC[alpha_idx]*np.ones(np.shape(X_data_alpha[:,alpha_idx])).ravel(), '*')
