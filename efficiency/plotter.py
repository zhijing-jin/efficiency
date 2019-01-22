# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import numpy as np
import itertools

from collections import Counter, defaultdict


class Plotter(object):

    def __init__(self, on_server=False, save_to='./img.png'):
        import socket

        self.on_server = on_server
        self.dir, self.default_img_name = save_to.rsplit('/', 1)

        if not socket.gethostname().endswith('.local'):
            if not on_server:
                print('[Server Mode] Images will be saved as files.')
                self.on_server = True

        import matplotlib
        if self.on_server:
            matplotlib.use('Agg')
            print('[Server Mode] Feel free to check the pics by: `python -m http.server 7002` ')
        else:
            matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        # plt.style.use('ggplot')

        tableau20 = [(31, 119, 180), (255, 127, 14), (255, 187, 120),
                     (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                     (148, 103, 189), (197, 176,
                                       213), (140, 86, 75), (196, 156, 148),
                     (227, 119, 194), (247, 182, 210), (174, 199,
                                                        232), (127, 127, 127), (199, 199, 199),
                     (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

        self.tableau20 = [(r / 255., g / 255., b / 255.)
                          for r, g, b in tableau20]

    def plot_hist(self, cnt, image_name='', width=0.7, title='', rotation=0):
        import matplotlib.pyplot as plt

        cnt_sorted = sorted(list(cnt.items()))

        xticks, values = zip(*cnt_sorted)
        x_axis = np.arange(len(xticks))

        fig, ax = plt.subplots()
        ax.bar(x_axis, values, width)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(xticks, rotation=rotation)
        ax.set_title(title)

        self._save_img(fig, plt, image_name)

    def plot_line(self, data, title='Age and death_rate', text=[]):
        import matplotlib.pyplot as plt

        item, val = zip(*data)
        fig, ax = plt.subplots()

        ax.plot(item, val, label='data')
        ax.legend(loc='upper right', frameon=True)
        ax.set_ylabel('Death rate')
        ax.set_xlabel('Age')
        ax.set_title(title)

        if text:
            for t, x, y in zip(text, item, val):
                ax.annotate(t, (x, y))
        plt.show()

    def plot_bar_stack(self, cnt=None, title='', xticks=[], cnt_categories=[], same_color=True):
        import matplotlib.pyplot as plt
        import pandas as pd

        pd_list = defaultdict(list)
        for cnt_type, cnt_lists in cnt.items():
            accul_list = np.zeros_like(cnt_lists[0])

            for cnt_list_i, cnt_list in enumerate(cnt_lists):
                accul_list = np.array(cnt_list) + accul_list
                pd_list[cnt_type] += [accul_list]

        x_axis = range(len(pd_list['before'][0]))

        # fig = plt.figure(figsize=(20, 10))
        fig, ax = plt.subplots()

        before_bar_list = [ax.bar(x_axis, ls, align='edge', width=-0.3, color=color)
                           for ls, color in zip(pd_list['before'][::-1], self.tableau20)]
        if same_color:
            after_bar_list = [ax.bar(x_axis, ls, align='edge', width=0.3, color=color)
                              for ls, color in zip(pd_list['after'][::-1], self.tableau20[::2])]
            legend_var = (ls[0] for ls in before_bar_list)
            ax.legend(legend_var, cnt_categories)
        else:
            after_bar_list = [ax.bar(x_axis, ls, align='edge', width=0.3, color=color)
                              for ls, color in zip(pd_list['after'][::-1], self.tableau20[1::2])]
            legend_names = ('{} ({})'.format(cate, cnt_type)
                            for cnt_type in ['before', 'after'] for cate in cnt_categories)
            legend_var = (ls[0] for bar_list in [
                before_bar_list, after_bar_list] for ls in bar_list)
            ax.legend(legend_var, legend_names)

        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks, rotation=90)

        ax.set_title(title)
        plt.show()

        import pdb
        pdb.set_trace()

    def plot_bar(self, cnt=None, title=''):
        import matplotlib.pyplot as plt

        import pandas as pd

        item_b, val_b = zip(*cnt['before'][0])
        item_a, val_a = zip(*cnt['after'][0])

        assert item_b == item_a

        df = pd.DataFrame({"date_x": ['before'] * len(item_b),
                           "Occurance_x": val_b,
                           "VTM_NM": item_b,
                           "date_y": ['after'] * len(item_b),
                           "Occurance_y": val_a})

        ax = df[["VTM_NM", "Occurance_x", "Occurance_y"]].plot(x='VTM_NM',
                                                               kind='bar',
                                                               # color=["r", "b"],
                                                               rot=90)
        ax.legend(["before", "after"])
        ax.set_title(title)
        plt.show()

        return
        import seaborn as sns

        data = {'item': ['red', 'green', 'blue'], 'val': [1, 2, 3]}
        if cnt:
            item, val = zip(*cnt)
            data = {'item': item,
                    'val': val}
        df = pd.DataFrame(data)
        ax = sns.barplot(x='item', y='val',
                         data=df,
                         color='black')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        ax.set(xlabel='common xlabel', ylabel='common ylabel')
        plt.ylim(0, 200)
        plt.show()

    def plot_heatmap(self, matrix=None, xticks=[], yticks=[], xlabel='', ylabel='', title='', image_name="", decimals=1,
                     annt_clr=None, cbarlabel="", show_img=False, **kwargs):
        import matplotlib.pyplot as plt

        if (not (type(matrix) is np.ndarray)) and (not xticks) and (not yticks):
            yticks = ["cucumber", "tomato", "lettuce", "asparagus",
                      "potato", "wheat", "barley"]
            xticks = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
                      "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

            matrix = np.array([[0.823, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                               [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                               [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                               [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                               [0.7, 1.7, 0.624, 2.6, 2.2, 6.2, 0.0],
                               [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                               [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
            title = "Harvest of local xticks (in tons/year)"

        matrix = np.around(matrix, decimals=decimals)

        fig, ax = plt.subplots()

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap='OrRd')
        # im = ax.imshow(matrix, vmin=0, vmax=1)

        # Create colorbar
        if cbarlabel:
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_yticks(np.arange(len(yticks)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        if annt_clr:
            for i in range(len(yticks)):
                for j in range(len(xticks)):
                    if matrix[i, j] != 0:
                        text = ax.text(
                            j, i, matrix[i, j], ha="center", va="center", color=annt_clr)

        # data = DataFrame(data=matrix, columns=xticks, index=yticks)
        # data.columns.name = xlabel
        # data.index.name = ylabel
        # pdb.set_trace()

        # ax = seaborn.heatmap(data)

        X = self._save_img(fig, plt, image_name)

        # plt.imshow(X, interpolation="none")
        # plt.show()

        return X

    def _save_img(self, fig, plt, image_name=''):
        res = None
        image_name = image_name if image_name else self.default_img_name

        if not self.on_server:
            plt.show()

        if image_name:
            fig.tight_layout()
            fig.canvas.draw()

            X = np.array(fig.canvas.renderer._renderer)
            # X = 0.2989 * X[:, :, 1] + 0.5870 * X[:, :, 2] + 0.1140 * X[:, :, 3] #
            # convert to black and white image

            from PIL import Image
            X = X[:, :, :3] if image_name.endswith(".jpg") else X
            im = Image.fromarray(X)
            file = os.path.join(self.dir, image_name)
            im.save(file)

            res = X

        plt.close('all')
        return res
