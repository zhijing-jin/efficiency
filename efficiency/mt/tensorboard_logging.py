# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from efficiency.log import fwrite
import os
import pdb
import random
import json


class TensorboardVisualizer(object, TensorboardLossRecord):

    def __init__(self, if_tensorboard, tensorboard_dir):
        super(TensorboardVisualizer, self).__init__()


class TensorboardLossRecord(object):
    """docstring for TensorboardLossRecord"""

    def __init__(self, if_tensorboard, tensorboard_dir):
        super(TensorboardLossRecord, self).__init__()
        self.if_tensorboard = if_tensorboard
        self.tensorboard_dir = tensorboard_dir
        if if_tensorboard:
            from tensorboardX import SummaryWriter

            self.w = SummaryWriter(log_dir=tensorboard_dir)

    def plot_loss(self, epoch, tl, va):

        if self.if_tensorboard:

            self.w.add_scalars('data/loss',
                               {'train': tl}, epoch)
            self.w.add_scalars('data/accuracy',
                               {'valid': va}, epoch)

    def plot_text(self, epoch, text, title='Text'):
        if self.if_tensorboard:

            self.w.add_text(
                title, text, epoch)

    def plot_img(self, epoch, img, caption='Attention', att_arr=None, image_name=''):
        if self.if_tensorboard:
            if att_arr is not None:
                import torch
                if type(att_arr) is torch.Tensor:
                    att_arr = att_arr.detach().cpu().numpy()
                elif type(att_arr) is np.ndarray:
                    pass
                else:
                    assert False, "type of att_arr must be either numpy.ndarray or torch.Tensor"

                img = heatmap2np(matrix=att_arr, xticks=[], yticks=[], xlabel='', ylabel='',
                                 title=json.dumps(caption[:30]), image_name=image_name, decimals=8, annt_clr='b', on_server=True)  # (480, 640, 4)
            try:
                self.w.add_text(add_image(
                    caption, img, epoch))
            except:
                img = np.transpose(img, (2, 0, 1))
                self.w.add_text(add_image(
                    caption, img, epoch))

    def close(self):
        if self.if_tensorboard:
            self.w.close()

    def visdo(self):
        from visdom import Visdom
        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector
        pass


def heatmap2np(matrix=None, xticks=[], yticks=[], xlabel='', ylabel='', title='', image_name="", decimals=1, annt_clr=None, on_server=True):
    import matplotlib
    if on_server:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    import seaborn

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

    im = ax.imshow(matrix, cmap='OrRd')

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
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if annt_clr:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                text = ax.text(
                    j, i, matrix[i, j], ha="center", va="center", color=annt_clr)

    # data = DataFrame(data=matrix, columns=xticks, index=yticks)
    # data.columns.name = xlabel
    # data.index.name = ylabel
    # pdb.set_trace()

    # ax = seaborn.heatmap(data)

    fig.tight_layout()
    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    # X = 0.2989 * X[:, :, 1] + 0.5870 * X[:, :, 2] + 0.1140 * X[:, :, 3] #
    # convert to black and white image

    if image_name:
        from PIL import Image
        X = X[:, :, :3] if image_name[-len(".jpg"):] == ".jpg" else X
        im = Image.fromarray(X)
        im.save(image_name)

    # plt.imshow(X, interpolation="none")
    # plt.show()
    plt.close('all')

    return X

    n = 3
    domain_size = 20

    x = np.random.randint(0, domain_size, (n, 2))

    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    ax.scatter(x[:, 0], x[:, 1], c="black", s=200, marker="*")
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    fig.add_axes(ax)

    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    X = 0.2989 * X[:, :, 1] + 0.5870 * X[:, :, 2] + 0.1140 * X[:, :, 3]
    show_var(["X", "x"])

    plt.imshow(X, interpolation="none", cmap="gray")
    plt.show()
