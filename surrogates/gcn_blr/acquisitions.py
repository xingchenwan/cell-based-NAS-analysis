import numpy as np
from scipy.special import erfc
from .utils import config_to_feature_adj

def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)

class EI(object):
    def __init__(self, surrogate_model, in_fill: str = 'best'):
        """
        Expected improvement
        """
        assert in_fill in ['best', 'posterior']
        self.in_fill = in_fill
        self.surrogate_model = surrogate_model

    def eval(self, X_list, Adj_list):
        """
        Return the negative expected improvement at the query point x2
        """

        mu, var = self.surrogate_model.predict(X_list, Adj_list)
        s = np.sqrt(var)
        fmin = self._get_incumbent()
        phi, Phi, u = get_quantiles(0, fmin, mu, s)
        f_acqu = s * (u * Phi + phi)

        return - f_acqu

    def _get_incumbent(self,):
        """
        Get the incumbent minimum
        """
        if self.in_fill == 'best':
            return self.surrogate_model.y_min
        else:
            X_list = self.surrogate_model.X_list
            adj_list = self.surrogate_model.adj_list

            mu_train, _ = self.surrogate_model.predict(X_list, adj_list)
            return np.min(mu_train)

    def optimise(self, candidates, top_n=5):
        """top_n: return the top n candidates wrt the acquisition function."""

        X_list, adj_list, valid_config_list = candidates
        acq_value_all_candidates = self.eval(X_list, adj_list)

        top_candidate_indices = np.argsort(acq_value_all_candidates)[:top_n]
        next_arch_config_list = [valid_config_list[idx]for idx in top_candidate_indices]
        next_X_list = [X_list[idx] for idx in top_candidate_indices]
        next_adj_list = [adj_list[idx] for idx in top_candidate_indices]
        acq_scores = acq_value_all_candidates[top_candidate_indices]

        return next_X_list, next_adj_list, next_arch_config_list, acq_scores

