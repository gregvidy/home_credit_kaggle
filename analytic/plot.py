from matplotlib import gridspec
from .auc import get_auc
from .ks import get_ks
from .frt import fraud_ratio_table
import numpy as np
import matplotlib.pyplot as plt
from .chimerge import get_information_value
import pandas as pd


def plot_triple(y_hat, y, section_n=10, name='', table=False):
    plt.subplots(1, 3, figsize=(12, 2.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])
    axes = [plt.subplot(gs[i]) for i in range(3)]

    get_auc(y_hat, y, name, axes[0])
    get_ks(y_hat, y, name, axes[1])
    res = fraud_ratio_table(y_hat, y, axes[2], section_n, table=table)
    if table:
        return res


def show_ks(data1, data2, name1='', name2=''):
    plt.subplots(1, 3, figsize=(12, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax = [plt.subplot(gs[i]) for i in range(3)]
    y1, y2 = [], []
    ecdf_min = min(min(data1), min(data2))
    ecdf_max = max(max(data1), max(data2))
    ecdf_x = np.arange(ecdf_min, ecdf_max, (ecdf_max - ecdf_min) / 100.0)
    for k in ecdf_x:
        y1.append((pd.Series(data1) <= k).sum() / len(data1))
        y2.append((pd.Series(data2) <= k).sum() / len(data2))

    res = abs(pd.Series(y1) - pd.Series(y2)).max()

    ax[0].hist(data1, density=True)
    ax[0].set_title('{} density'.format(name1).strip())

    ax[1].hist(data2, density=True)
    ax[1].set_title('{} density'.format(name2).strip())

    ax[2].plot(ecdf_x, y1)
    ax[2].plot(ecdf_x, y2)
    ax[2].set_ylim([-0.03, 1.03])
    ax[2].set_title('{}-{} ks: {:.4f}'.format(name1, name2, res).strip())

    return res


def iv_woe_plot(_x, _y, method='chimerge', max_cat=7, initial_intervals=20, ax=None):
    iv_list, woe, bins, weight, gcl, bcl = get_information_value(_x, _y, method, max_cat, initial_intervals)
    iv = np.sum(iv_list)
    if ax is None:
        ax = plt.figure(figsize=(6.3, 5)).gca()
    if ax is not False:
        data = [bcl, gcl]
        columns = ['({},{}]'.format(np.round(bins[i], 2), np.round(bins[i + 1], 2)) for i in range(len(bins) - 1)]
        columns[0] = '[' + columns[0][1:]
        rows = ['True', 'False', 'Bad Rate', 'WOE', 'IV']
        colors = ['orangered', '#AADC3D', '#00000000', '#00000000', '#00000000']
        alphas = [0.8, 0.9]
        n_rows = len(data)

        index = np.arange(len(columns)) + 0.2
        bar_width = 0.5

        y_offset = np.zeros(len(columns))

        cell_text = []

        for row in range(n_rows):
            ax.bar(
                index,
                data[row],
                bar_width,
                bottom=y_offset,
                color=colors[row],
                alpha=alphas[row]
            )
            y_offset = y_offset + data[row]
            cell_text.append(['%1d' % (x) for x in y_offset])

        default_rate = list(map(lambda x: x[0] / (x[0] + x[1]), zip(bcl, gcl)))
        cell_text.append(list(map(lambda x: np.round(x, 3), default_rate)))
        cell_text.append(list(map(lambda x: np.round(x, 3), woe)))
        cell_text.append(list(map(lambda x: np.round(x, 3), iv_list)))

        # Add a table at the bottom of the axes
        the_table = ax.table(cellText=cell_text,
                             rowLabels=rows,
                             rowColours=colors,
                             colLabels=columns,
                             loc='bottom',
                             cellLoc='center',
                             rowLoc='center')
        the_table.fontsize = 10
        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.9, top=0.9)

        ax.set_xticks([])
        ax.set_ylabel('Count')
        ax.set_title('IV: {:.6}'.format(iv))

        ax_pr = ax.twinx()
        ax_pr.plot(index,
                   default_rate,
                   linestyle='-',
                   color='dodgerblue',
                   marker='o',
                   markeredgecolor='red',
                   markeredgewidth=1,
                   markerfacecolor='snow'
                   )
        plt.show()
    else:
        return iv


if __name__ == '__main__':
    pass
