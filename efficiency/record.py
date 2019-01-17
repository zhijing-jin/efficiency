# -*- coding: utf-8 -*-
from __future__ import division, print_function


class NoteTaker(object):

    def __init__(self, file):
        from efficiency.log import fwrite
        self.file = file
        fwrite('', file)

    def print(self, expression):
        print(expression)
        fwrite(expression + '\n', self.file, mode='a')
