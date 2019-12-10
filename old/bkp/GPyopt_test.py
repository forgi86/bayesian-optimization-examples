import GPyOpt
import GPy
import GPyOpt.experiment_design
import numpy as np


if __name__ == '__main__':

    func  = GPyOpt.objective_examples.experiments2d.branin()
    func.plot()

    # objective function (single objective)
    objective = GPyOpt.core.task.SingleObjective(func.f)

    space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},
                                        {'name': 'var_2', 'type': 'continuous', 'domain': (1,15)}])


    # objective model (GP)
    model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)

    # optimizer
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

    # initial points
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

    # acquisition function
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, space, optimizer=aquisition_optimizer)

    # sequential evaluator (standard)
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # modular optimizer
    myBopt = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition,
                                                        evaluator, initial_design)

    # Run the optimization
    max_iter = 10     # evaluation budget
    max_time = np.inf     # time budget
    eps      = 10e-8  # Minimum allows distance between the las two observations
    myBopt.run_optimization(max_iter=max_iter, max_time=max_time, eps=eps, verbosity=False, context=None)

    myBopt.plot_acquisition()
    myBopt.plot_convergence()


