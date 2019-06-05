import numpy as np
from scipy.stats import norm
import GPyOpt
from GPyOpt.experiment_design import initial_design
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


if __name__ == '__main__':

    n_iter = 10
    ALPHA_VEC = np.arange(0, 1.1, 0.1)
    n_alpha = len(ALPHA_VEC)

#    bounds = np.array([[-3.0, 3.0]])
    bounds =[{'name': 'x', 'type': 'continuous', 'domain': (-3, 3)}]

    # Gaussian process with Matérn kernel as surrogate model
    X_init = np.array([[-2.],
                       [-1.],
                       [0.],
                       [1.],
                       [2.]
                      ]
                     )
    n_init = np.shape(X_init)[0]
    J_init = f_vec(X_init)

    X_sample = np.copy(X_init)
    J_sample = np.copy(J_init)

    BO_LIST = []
    # Run the optimization
    for iter_idx in range(n_iter):
        for alpha_idx in range(n_alpha):
            alpha = ALPHA_VEC[alpha_idx]

            # Prepare data for next optimization
            J_sample_alpha = f_tradeoff(J_sample, alpha)


            def f_opt(x):
                return f_tradeoff(f_vec(x), alpha)


            myBopt = GPyOpt.methods.BayesianOptimization(f=f_opt,  # function to optimize
                                                         domain=bounds,  # box-constraints of the problem
                                                         acquisition_type='EI',  # expected improvement
                                                         exact_feval=False,
                                                         X=X_sample,
                                                         Y=J_sample_alpha)  # Initial design matrix
            X_next = myBopt.suggest_next_locations()

            # Compute vector output for next sample
            J_next = f_vec(X_next)

            X_sample = np.vstack([X_sample,
                                  X_next])

            J_sample = np.vstack([J_sample,
                                  J_next])

            if iter_idx == n_iter - 1:
                BO_LIST.append(myBopt)


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
        Y_tradeoff = f_tradeoff(Y_vec,alpha)
        Jj[idx_alpha, :] = Y_tradeoff.ravel()

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
