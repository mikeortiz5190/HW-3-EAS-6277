


if __name__ == '__main__':
    import numpy as np
    # from numpy.random import normal
    from numpy import pi
    from numpy import cos
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal
    from matplotlib import style
    style.use('fivethirtyeight')



    '''Question 2'''

    ################# FAILURE FUNCTION ######################

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from scipy.stats import multivariate_normal
    from numpy import sqrt

    # Sample parameters
    mu = np.array([2, 8])
    sigma = np.array([[1, 0.5], [0.5, 1]])
    rv = multivariate_normal(mu, sigma)
    sample = rv.rvs(1000)

    total_samples = 1000000

    x1 = np.random.normal(mu[0], sqrt(sigma[0][0]), total_samples)
    x2 = np.random.normal(mu[1], sqrt(sigma[1][1]), total_samples)


    def g_of_x1_x1(X1, X2):
        failure = 0
        success = 0
        mean = []
        for i in range(total_samples):
            g = (X2[i] - (5.1 / (4 * pi) * X2[i] ** 2) + (5 * X1[i] / pi) - 6) ** 2 + 10 * (1 - (1 / (8 * pi))) * cos(
                X1[i]) + 10
            mean.append(g)
            if g > 50:
                failure = failure + 1
            elif g < 50:
                success = success + 1
        mean = np.mean(mean)
        total_prob_of_failure = failure / (failure + success)
        #total_prob_of_failure = "{:.2f}".format(total_prob_of_failure)

        return [total_prob_of_failure, mean, failure, success]


    solution_two = g_of_x1_x1(x1, x2)

    print()
    print(f'The probobility of Failure is {solution_two[0]} using {total_samples} samples.\n')
    print(f'The mean failure function output: {solution_two[1]}\n')
    print(f'Number of succesful trials: {solution_two[3]}\n')
    print(f'Number of failed trials: {solution_two[2]}\n')

    #################################################
    #################### Part B - binomial Plot ####################
    #################################################

    from scipy.stats import binom
    import matplotlib.pyplot as plt
    import math

    # setting the values
    # of n and p
    n = 500

    p = solution_two[0]
    # defining list of r values
    r_values = list(range(n + 1))
    # list of pmf values
    dist = [binom.pmf(r, n, p) for r in r_values]
    #print(dist)
    max_value = max(dist)
    index_max = dist.index(max_value)
    print(f'{index_max} failures has a probobility of {max_value}\n')
    ub = math.floor(p + (1.96 * (p * (1-p) / n))**0.5)
    lb = p - (1.96 * (p * (1-p) / n))**0.5
    confidence_interval_95 = [lb, ub]
    print(f'the 95% confidence interval for failure is: {confidence_interval_95}\n')
    # plotting the graph
    plt.bar(r_values, dist)
    plt.title(f'BD given a failure probobility of {p}')
    plt.show()

    #################################################
    #################### 3D Plot ####################
    #################################################
    # Bounds parameters
    '''
    x_abs = 10
    y_abs = 10
    x_grid, y_grid = np.mgrid[-x_abs:x_abs:.02, -y_abs:y_abs:.02]

    pos = np.empty(x_grid.shape + (2,))
    pos[:, :, 0] = x_grid
    pos[:, :, 1] = y_grid

    levels = np.linspace(0, 1, 40)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Removes the grey panes in 3d plots
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # The heatmap
    ax.contourf(x_grid, y_grid, 0.1 * rv.pdf(pos),
                zdir='z', levels=0.1 * levels, alpha=0.9)

    # The wireframe
    ax.plot_wireframe(x_grid, y_grid, rv.pdf(
        pos), rstride=100, cstride=100, color='k')

    # The scatter. Note that the altitude is defined based on the pdf of the
    # random variable
    ax.scatter(sample[:, 0], sample[:, 1], 1.05 * rv.pdf(sample), c='k')

    ax.legend()
    ax.set_title("Multivariate Normal distribution")
    ax.set_xlim3d(-x_abs, x_abs)
    ax.set_ylim3d(-y_abs, y_abs)
    ax.set_zlim3d(0, 1)
    plt.show()
'''