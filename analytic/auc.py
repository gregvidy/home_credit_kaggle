from .draw_utils import type_check, get_tp_fp
from sklearn import metrics


def get_auc(y_hat, y, name='', ax=None):
    y_hat = type_check(y_hat)
    y = type_check(y)

    if len(y_hat) != len(y):
        raise Exception('y_hat and y is supposed to have same length')
        return None

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat, pos_label=0)

    auc = round(metrics.auc(fpr, tpr), 5)
    if ax is not None:
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], 'k-.')
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title('{} AUC: {}'.format(name, auc).strip())

    return auc


if __name__ == '__main__':
    pass
