import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def f(x):
    return sin(x)*math.exp(-x*x/16) 

def make_data(N, mean, var, nb_irr, nb_ls):
    min_bound = -10
    max_bound = 10
    ls = []

    for i in range(nb_ls):
        x_tab = np.random.uniform(min_bound, max_bound, N) #the relevant points

        if not nb_irr == 0:
            x_tab.append(np.random.uniform(min_bound, max_bound, nb_irr)) #the irrelevant points

        x_tab.sort()
        y_tab = [f(x) + np.random.normal(mean, math.sqrt(var))/10 for x in x_tab]

        ls.append((x,y)) #crasset avait fais un truc chelou avec le tableau de x
    return ls

def to_fit(learning_samples, regression_method, nb_neighbors=None):
    fitted = []

    for ls in learning_samples:
        (x_samples, y_samples) = ls
        if regression_method == "KNR":
            fitted.append(KNeighborsRegressor(nb_neighbors).fit(x_samples, y_samples))
        else:
            fitted.append(LinearRegression().fit(x_samples, y_samples))

    return fitted

def make_plot(x, x_label, y1, y1_label, y2, y2_label, y3, y3_label, y4, y4_label, filename):
    plt.figure()

    plt.plot(x, y1, label=y1_label)
    plt.plot(x, y2, label=y2_label)
    plt.plot(x, y3, label=y3_label)
    plt.plot(x, y4, label=y4_label)

    plt.xlabel(x_label)
    plt.legend(loc="upper right")

    plt.savefig(filename)

def compute_residual_error(elm_to_test, mean, var, nb_ls): #var_y {y}
    y_tab = []
    for i in range(nb_ls):
        y_tab.append(f(elm_to_test) + np.random.normal(mean, math.sqrt(var))/10)

    return np.var(y)

def compute_squared_bias(elm_to_test, fitted_models): # bias^2 = error between bayes and average model (over all LS)
    predicted = []
    for model in fitted_models:
        predicted.append(model.predict([[elm_to_test]]))
    average_model = np.mean(predicted)

    return (average_model- f(elm_to_test))**2 #^2

def compute_variance_ls(elm_to_test, fitted_models): #var_LS {y}
    predicted = []
    for model in fitted_models:
        predicted.append(model.predict([[elm_to_test]])) # car pour l'arg de predict X : array-like, shape (n_query, n_features), or (n_query, n_indexed) if metric == ‘precomputed’

    return np.var(predicted)

def compute_expected_error(residual_error, squared_bias, variance_ls):
    return residual_error + squared_bias + variance_ls

def Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors=None):
    ls = make_data(N, mean, var, nb_irr, nb_ls)
    fitted_models = to_fit(ls, regression_method, nb_neighbors)

    residual_errors = []
    squared_bias = []
    variances_ls = []
    expected_errors = []

    for test in test_set:
        residual_error = compute_residual_error(test, mean, var, nb_ls)
        squared_bias_val = compute_squared_bias(test, fitted_models)
        variance_ls = compute_variance_ls(test, fitted_models)
        expected_error = compute_expected_error(residual_error, squared_bias_val, variance_ls)

        residual_errors.append(residual_error)
        squared_bias.append(squared_bias_val)
        variances_ls.append(variance_ls)
        expected_errors.append(expected_error)

    return residual_errors, squared_bias, variances_ls, expected_errors

if __name__ == "__main__":
    N = 100
    nb_ls = 5
    test_set = np.arrange(-10,10,0.01)
    mean = 0
    var = 1
    nb_irr = 0
    nb_neighbors = 5

    to_compute = "Q_3d"
    #to_compute = "change_size_ls"
    #to_compute = "change_complexity"
    #to_compute = "change_nb_irrelevant"

    if to_compute == "Q_3d":
        residual_errors, squared_bias, variances_ls, expected_errors = Q_3d(N, nb_irr, nb_ls, mean, var, test_set, "KNR", nb_neighbors)
        make_plot(test_set, "x", residual_errors, "Residual error", squared_bias, "Squared bias", variances_ls, "Variance", expected_errors, "Expected errors", "KNR_3d.png")

        residual_errors, squared_bias, variances_ls, expected_errors = Q_3d(N, nb_irr, nb_ls, mean, var, test_set, "LNR")
        make_plot(test_set, "x", residual_errors, "Residual error", squared_bias, "Squared bias", variances_ls, "Variance", expected_errors, "Expected errors", "LNR_3d.png")

    elif to_compute == "change_size_ls":

    elif to_compute == "change_complexity":

    else:
