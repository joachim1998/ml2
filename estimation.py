import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.utils import check_random_state
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def f(x):
    return math.sin(x)*math.exp(-x*x/16)

def get_y_tuple(x_tuple, mean, var):
    y = []
    for x in x_tuple:
        y.append(f(x) + np.random.normal(mean, math.sqrt(var))/10)

    return tuple(y)

def make_data(N, mean, var, nb_irr, nb_ls):
    min_bound = -10
    max_bound = 10
    ls = []

    for i in range(nb_ls):
        x_tab = np.random.uniform(min_bound, max_bound, N).tolist() #the relevant variable
        
    if nb_irr != 0:
        for i in range(nb_irr)
                x_irr.append(np.random.uniform(min_bound, max_bound, N).tolist()) #the irrelevant variables

        #if not nb_irr == 0:
            #for j in range(len(x_tab)):
                #x_tab[j] = (x_tab[j],) + tuple(np.random.uniform(min_bound, max_bound, nb_irr)) #the irrelevant points
               # x_tab[j].append(np.random.uniform(min_bound, max_bound, nb_irr).tolist())

        #x_tab.sort()
        y_tab = []
        x_array = []

        for x in x_tab:
            y_tab.append(f(x) + np.random.normal(mean, math.sqrt(var))/10)
            x_array.append([x,x_irr])

        ls.append((x_array,y_tab))
    return ls

def to_fit(learning_samples, regression_method, nb_neighbors=None):
    fitted = []

    for ls in learning_samples:
        (x_samples, y_samples) = ls
        print(x_samples)
        print(y_samples)
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
    plt.ylabel("Error")
    plt.legend(loc="upper right")

    plt.savefig(filename)

def compute_errors(elm_to_test, mean, var, nb_ls, fitted_models):
    #residual error
    y_tab = []
    for i in range(nb_ls):
        y_tab.append(f(elm_to_test) + np.random.normal(mean, math.sqrt(var))/10)
    residual_error = np.var(y_tab)

    #squared_bias and variance LS
    predicted = []
    for model in fitted_models:
        predicted.append(model.predict([[elm_to_test]]))
    variance_ls = np.var(predicted)
    squared_bias = (np.mean(predicted) - f(elm_to_test))**2

    #expected error
    expected_error = residual_error + squared_bias + variance_ls

    return residual_error, squared_bias, variance_ls, expected_error

def Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors=None):
    ls = make_data(N, mean, var, nb_irr, nb_ls)
    fitted_models = to_fit(ls, regression_method, nb_neighbors)

    residual_errors = []
    squared_bias = []
    variances_ls = []
    expected_errors = []

    for test in test_set:
        residual_error, squared_bias_val, variance_ls, expected_error = compute_errors(test, mean, var, nb_ls, fitted_models)

        residual_errors.append(residual_error)
        squared_bias.append(squared_bias_val)
        variances_ls.append(variance_ls)
        expected_errors.append(expected_error)

    return residual_errors, squared_bias, variances_ls, expected_errors

def mean_Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors=None):
    elements = []

    for elm in Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors):
        elements.append(np.mean(elm))

    return elements

def change_size_ls(nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors=None):
    size_ls = range(10,200,1)

    residual_errors = []
    squared_bias = []
    variances_ls = []
    expected_errors = []

    for N in size_ls:
        residual_error, squared_bias_val, variance_ls, expected_error = mean_Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors)

        residual_errors.append(residual_error)
        squared_bias.append(squared_bias_val)
        variances_ls.append(variance_ls)
        expected_errors.append(expected_error)

    return size_ls, residual_errors, squared_bias, variances_ls, expected_errors


def change_complexity(N, nb_irr, nb_ls, mean, var, test_set, regression_method):
    complexity = range(1,15,1)

    residual_errors = []
    squared_bias = []
    variances_ls = []
    expected_errors = []

    for nb_neighbors in complexity:
        residual_error, squared_bias_val, variance_ls, expected_error = mean_Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors)

        residual_errors.append(residual_error)
        squared_bias.append(squared_bias_val)
        variances_ls.append(variance_ls)
        expected_errors.append(expected_error)

    return complexity, residual_errors, squared_bias, variances_ls, expected_errors

def change_nb_irrelevant(N, nb_ls, mean, var, test_set, regression_method, nb_neighbors=None):
    nb_irrelevant = range(0,100,1)

    residual_errors = []
    squared_bias = []
    variances_ls = []
    expected_errors = []

    for nb_irr in nb_irrelevant:
        residual_error, squared_bias_val, variance_ls, expected_error = mean_Q_3d(N, nb_irr, nb_ls, mean, var, test_set, regression_method, nb_neighbors)

        print("OK pour nb_irre = " + str(nb_irr))

        residual_errors.append(residual_error)
        squared_bias.append(squared_bias_val)
        variances_ls.append(variance_ls)
        expected_errors.append(expected_error)

    return nb_irrelevant, residual_errors, squared_bias, variances_ls, expected_errors


if __name__ == "__main__":
    N = 100
    nb_ls = 5
    test_set = np.arange(-10,10,0.01)
    mean = 0
    var = 1
    nb_irr = 0
    nb_neighbors = 5

    #to_compute = "Q_3d"
    #to_compute = "change_size_ls"
    #to_compute = "change_complexity"
    to_compute = "change_nb_irrelevant"

    if to_compute == "Q_3d":
        residual_errors, squared_bias, variances_ls, expected_errors = Q_3d(N, nb_irr, nb_ls, mean, var, test_set, "KNR", nb_neighbors)
        make_plot(test_set, "x", residual_errors, "Residual error", squared_bias, "Squared bias", variances_ls, "Variance", \
        		 expected_errors, "Expected errors", "KNR_3d.png")

        residual_errors, squared_bias, variances_ls, expected_errors = Q_3d(N, nb_irr, nb_ls, mean, var, test_set, "LNR")
        make_plot(test_set, "x", residual_errors, "Residual error", squared_bias, "Squared bias", variances_ls, "Variance", \
        		 expected_errors, "Expected errors", "LNR_3d.png")

    elif to_compute == "change_size_ls":
        size_ls, residual_errors, squared_bias, variances_ls, expected_errors = change_size_ls(nb_irr, nb_ls, mean, var, test_set, "KNR", nb_neighbors)
        make_plot(size_ls, "Size of the learning set", residual_errors, "Mean residual error", squared_bias, "Mean squared bias", \
        		 variances_ls, "Mean variance", expected_errors, "Mean expected errors", "KNR_change_size.png")

        size_ls, residual_errors, squared_bias, variances_ls, expected_errors = change_size_ls(nb_irr, nb_ls, mean, var, test_set, "LNR")
        make_plot(size_ls, "Size of the learning set", residual_errors, "Mean residual error", squared_bias, "Mean squared bias", \
        		 variances_ls, "Mean variance", expected_errors, "Mean expected errors", "LNR_change_size.png")

    elif to_compute == "change_complexity":
        complexity, residual_errors, squared_bias, variances_ls, expected_errors = change_complexity(N, nb_irr, nb_ls, mean, var, test_set, "KNR")
        make_plot(complexity, "Complexity", residual_errors, "Mean residual error", squared_bias, "Mean squared bias", \
        		 variances_ls, "Mean variance", expected_errors, "Mean expected errors", "KNR_change_complexity.png")

    else:
        nb_irrelevant, residual_errors, squared_bias, variances_ls, expected_errors = change_nb_irrelevant(N, nb_ls, mean, var, test_set, "KNR", nb_neighbors)
        make_plot(nb_irrelevant, "Number of irrelevant variables", residual_errors, "Mean residual error", squared_bias, \
        		 "Mean squared bias", variances_ls, "Mean variance", expected_errors, "Mean expected errors", "KNR_change_irr.png")

        nb_irrelevant, residual_errors, squared_bias, variances_ls, expected_errors = change_nb_irrelevant(N, nb_ls, mean, var, test_set, "LNR")
        make_plot(nb_irrelevant, "Number of irrelevant variables", residual_errors, "Mean residual error", squared_bias, \
        		 "Mean squared bias", variances_ls, "Mean variance", expected_errors, "Mean expected errors", "LNR_change_irr.png")
