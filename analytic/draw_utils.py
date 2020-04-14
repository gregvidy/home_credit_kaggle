import pandas as pd
import numpy as np


def type_check(x):
    if isinstance(x, list):
        return np.array(x)
    elif (isinstance(x, pd.DataFrame)) or (isinstance(x, pd.Series)):
        return x.map(lambda _x: int(_x)).values
    else:
        return x


def get_tp_fp(y, y_hat, split_n, prob=False):
    data = pd.DataFrame([y_hat, y], index=['score', 'label']).T
    data['bin'] = pd.qcut(data.score, split_n, duplicates='drop')
    tmp = data.groupby(
        ['bin', 'label']
    ).count(
    ).reset_index(
    ).pivot(
        index='bin',
        columns='label',
        values='score'
    ).fillna(0)
    if not prob:
        return tmp
    else:
        prob = ((data.score.sort_values() - data.score.min()) / (data.score.max() - data.score.min())).quantile(
            np.arange(0, 1, 1 / split_n)
        )
    return tmp, prob
