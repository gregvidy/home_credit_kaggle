import numpy as np
import matplotlib.pyplot as plt
from ..analytic.chi_merge import get_information_value


def iv_woe_plot(_x, _y, method='chimerge', max_cat=7, ax=None):
    iv_list, woe, bins, weight, gcl, bcl = get_information_value(_x, _y, method, max_cat)
    iv = np.sum(iv_list)
    if ax is None:
        ax = plt.figure(figsize=(6.3, 5)).gca()
    if ax is not False:
        data = [bcl, gcl]
        columns = ['({},{}]'.format(np.round(bins[i], 2), np.round(bins[i + 1], 2)) for i in range(len(bins) - 1)]
        columns[0] = '[' + columns[0][1:]
        rows = ['True', 'False', 'Bad Rate', 'WOE', 'IV']

        colors = ['r', 'yellowgreen', '#00000000', '#00000000', '#00000000']
        alphas = [0.7, 1]
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
