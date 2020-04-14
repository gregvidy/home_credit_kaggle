import numpy as np
import pandas as pd


def chi_square(arr):
    group_count = np.sum(arr, axis=0, keepdims=True)
    label_ratio = np.sum(arr, axis=1, keepdims=True) / np.sum(group_count)
    expectation = np.matmul(label_ratio, group_count) + 0.0000000001
    return np.sum((arr - expectation) ** 2 / expectation)


def chi_merge(_x, _y, initial_intervals, target_intervals=5):
    index, bins = pd.qcut(_x, initial_intervals, labels=False, retbins=True, duplicates='drop')
    df_y = pd.get_dummies(_y).groupby(index).sum()
    _y = df_y.values

    while _y.shape[0] > target_intervals:
        chi2 = np.array([chi_square(_y[i:i + 2, :]) for i in range(_y.shape[0] - 1)])
        pos = np.argmin(chi2)
        _y[pos, :] += _y[pos + 1, :]
        _y = np.delete(_y, pos, axis=0)
        bins = np.delete(bins, pos + 1)

    index, bins = pd.cut(_x, bins, retbins=True, include_lowest=True, duplicates='drop')
    return index, bins


def make_binning(_x, bin_num, _y=None, method='linspace', initial_intervals=20):
    if isinstance(_x, pd.Series):
        _x = _x.values
    if method == 'linspace':
        bins = np.linspace(np.min(_x), np.max(_x), bin_num + 1)
        indxs = np.digitize(_x, bins)
        indxs[np.where(indxs == bin_num + 1)] = bin_num
    if method == 'quantile':
        indxs, bins = pd.qcut(_x, 6, retbins=True, duplicates='drop')
    if (method == 'chimerge') & (_y is not None):
        indxs, bins = chi_merge(_x, _y, initial_intervals, target_intervals=bin_num)
    return indxs, bins


def naive_woe(_x, _y, max_cat):
    levels, counts = np.unique(_x, return_counts=True)
    if len(levels) > max_cat:
        raise Exception('level should not more than {}'.format(max_cat))
    total = np.bincount(_y)
    woe_list = []
    weight_list = []
    good_count_list = []
    bad_count_list = []
    for level, count in zip(levels, counts):
        index = np.where(_x == level)[0]
        group = np.bincount(_y[index])
        p_y_0 = group[0] / total[0]
        p_y_1 = group[1] / total[1]
        woe_list.append(np.log(p_y_1 / p_y_0))
        weight_list.append(p_y_1 - p_y_0)
        good_count_list.append(group[0])
        bad_count_list.append(group[1])
    return woe_list, weight_list, good_count_list, bad_count_list


def get_woe(_x, _y, method, max_cat, details, initial_intervals):
    levels = np.unique(_x)
    if len(levels) > max_cat:
        _x, bins = make_binning(_x, max_cat, _y, method, initial_intervals)
    woe_list, weight_list, good_count_list, bad_count_list = naive_woe(_x, _y, max_cat)
    if details:
        return woe_list, bins, weight_list, good_count_list, bad_count_list
    else:
        return woe_list, bins


def get_information_value(_x, _y, method, max_cat, initial_intervals):
    woe, bins, weight, gcl, bcl = get_woe(_x, _y, method, max_cat, details=True, initial_intervals=initial_intervals)
    iv_list = list(map(lambda x: x[0] * x[1], zip(woe, weight)))
    return iv_list, woe, bins, weight, gcl, bcl
