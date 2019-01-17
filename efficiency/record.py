# -*- coding: utf-8 -*-
from __future__ import division, print_function
from efficiency.log import fwrite


class NoteTaker(object):

    def __init__(self, file):
        self.file = file
        self.content = []
        fwrite(self._text(), file)

    def _text(self):
        return '\n'.join(self.content)

    def print(self, *expressions):
        expression = ' '.join(str(e) for e in expressions)
        print(expression)

        self.content += [expression]
        fwrite(self._text(), self.file, mode='a')
