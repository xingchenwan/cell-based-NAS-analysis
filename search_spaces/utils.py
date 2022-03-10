import numpy as np


def snap_to_valid_anasod(x, summing_const, normalized=True):
    """
    Snap a possible ANASOD encoding to its nearest valid config using this method:
    https://link.springer.com/article/10.1007/s10898-013-0126-2

    x: the input candidate ANASOD encoding.
    summing_const: the summation constant that an unnormalised ANASOD encoding should sum to. For the cell-based NAS
        case, this should be the number of spots (OP_SPOTS) within the cell
    normalized: whether x is expected to sum to 1.
    """
    if normalized:
        assert np.isclose(np.sum(x), 1., rtol=0.01), np.sum(x)
        x_ = x * summing_const
    else:
        assert summing_const > len(x), "the summing constant must be larger than the number of elements, or " \
                           "turn probability_simplex=True for probability simplex rounding."
        x_ = np.copy(x)

    x_floored = np.floor(x_)
    x_fractional = x_ - x_floored
    # this should be an integer
    k = np.sum(x_fractional)
    # assert np.isclose(k, np.round(k), rtol=0.01), k
    k = int(np.round(k))
    # find the k largest indices
    all_indices = np.argpartition(x_fractional, -k)
    ceil_indices, floor_indices = all_indices[-k:], all_indices[:-k]
    x_[ceil_indices] = np.ceil(x_[ceil_indices])
    x_[floor_indices] = np.floor(x_[floor_indices])
    if normalized:
        x_ /= summing_const
    return x_


def is_upper_triangular(matrix):
    """True if matrix is 0 on diagonal and below."""
    for src in range(np.shape(matrix)[0]):
        for dst in range(0, src + 1):
            if matrix[src, dst] != 0:
                return False

    return True


def _check_data(pred, target):
    if not isinstance(pred, np.ndarray): pred = np.array(pred).reshape(-1)
    if not isinstance(target, np.ndarray): target = np.array(target).reshape(-1)
    return pred, target


def rmse(pred, target) -> float:
    """Compute the root mean squared error"""
    pred, target = _check_data(pred, target)
    assert pred.shape[0] == target.shape[0], 'predictant shape ' + \
                                             str(pred.shape[0]) + ' but target shape' + str(target.shape[0])
    n = pred.shape[0]
    return np.sqrt(np.sum((pred - target) ** 2) / n)


def nll(pred, pred_std,  target) -> float:
    """Compute the negative log-likelihood (over the validation dataset)"""
    from scipy.stats import norm
    pred, target = _check_data(pred, target)
    total_nll_origin = - np.mean(norm.logpdf(target, loc=pred, scale=pred_std))
    return total_nll_origin


def spearman(pred, target) -> float:
    """Compute the spearman correlation coefficient between prediction and target"""
    from scipy import stats
    pred, target = _check_data(pred, target)
    coef_val, p_val = stats.spearmanr(pred, target)
    return coef_val


def average_error(pred, target, log_val=True) -> float:
    pred, target = _check_data(pred, target)
    if log_val:
        pred = np.exp(pred)
        target = np.exp(target)
    return np.mean(np.abs((pred - target))).item()