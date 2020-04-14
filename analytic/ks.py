from .draw_utils import type_check
import pandas as pd
from sklearn import metrics
import numpy as np


def type_check(x):
    if isinstance(x, list):
        return np.array(x)
    elif (isinstance(x, pd.DataFrame)) or (isinstance(x, pd.Series)):
        return x.map(lambda _x: int(_x)).values
    else:
        return x


def get_ks(y_hat, y, name='', ax=None):
    y_hat = type_check(y_hat)
    y = type_check(y)

    if len(y_hat) != len(y):
        raise Exception('y_hat and y is supposed to have same length')
        return None
    if np.mean(y_hat) > 1:
        y_pos = y_hat[np.where(y == False)]
        y_neg = y_hat[np.where(y == True)]

        X1 = np.sort(y_pos)
        F1 = np.array(range(len(y_pos))) / float(len(y_pos))

        X2 = np.sort(y_neg)
        F2 = np.array(range(len(y_neg))) / float(len(y_neg))

    else:
        y_pos = y_hat[np.where(y == False)]
        y_neg = y_hat[np.where(y == True)]

        X1 = np.sort(y_pos)
        F1 = np.array(range(len(y_pos))) / float(len(y_pos))

        X2 = np.sort(y_neg)
        F2 = np.array(range(len(y_neg))) / float(len(y_neg))

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=0)
    ks = round(max(tpr - fpr), 5)

    if ax is not None:
        ax.plot(X1, F1)
        ax.plot(X2, F2)
        if np.mean(y_hat) < 0:
            ax.set_xlabel('Estimated Probability')
            ax.set_xlim([0, 1])
        else:
            ax.set_xlabel('Score Range')
        ax.set_ylabel('Actual Probability')
        ax.set_title('{} KS: {}'.format(name, ks).strip())
        ax.set_ylim([0.0, 1.01])

    return ks


if __name__ == '__main__':
    pass
