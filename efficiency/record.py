# -*- coding: utf-8 -*-
from __future__ import division, print_function
from efficiency.log import fwrite


class NoteTaker(object):

    def __init__(self, file):
        self.file = file
        fwrite('', file)

    def print(self, expression):
        print(expression)
        fwrite(expression + '\n', self.file, mode='a')
