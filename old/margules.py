import numpy as np
import scipy.constants as constants
import numpy as np
import scipy.optimize as opt
from sklearn import metrics


class Margules:
    def Ge(x, A):
        return A * x * (1 - x)

    def gamma1(x, A, T):
        return np.exp(A * (1 - x) ** 2 / (constants.R * T))

    def gamma2(x, A, T):
        return np.exp(A * x ** 2 / (constants.R * T))


def get_parameter(x, G_e):
    A, params_cov = opt.curve_fit(Margules.Ge, x, G_e, p0=1000, maxfev=10000)
    return A


def get_accuracy(G_e, Ge):
    return metrics.r2_score(G_e, Ge)
