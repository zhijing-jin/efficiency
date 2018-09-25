# -*- coding: utf-8 -*-
from __future__ import division, print_function
import pickle
import random
import numpy as np
import sys
import os.path
import socket


import pprint
import pdb


def show_var(expression):
    import json

    for i in expression:
        frame = sys._getframe(1)
        value = eval(i, frame.f_globals, frame.f_locals)

        if ' object at ' in repr(value):
            value = vars(value)
            value = json.dumps(value, indent=2)
            print(i, ':', value)
        else:
            print(i, ':', repr(value))


def write_var(var, path='data/debug/var'):
    with open(path, 'w') as f:
        f.write(path.split('/')[-1] +
                ' = ' + repr(var) + '\n')


def fwrite(new_doc, path):
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    with open(path, 'w') as f:
        f.write(new_doc)


def fread(path):
    with open(path, 'r') as f:
        return f.readlines()


def show_time(what_happens='', cat_server=False):
    import datetime

    disp = '⏰\ttime: ' + \
        datetime.datetime.now().strftime('%m%d%H%M-%S')
    disp = disp + '\t' + what_happens if what_happens else disp
    print(disp)
    curr_time = datetime.datetime.now().strftime('%m%d%H%M')

    if cat_server:
        hostname = socket.gethostname()
        prefix = "rosetta"
        if hostname.startswith(prefix):
            host_id = hostname[len(prefix):]
            try:
                host_id = int(host_id)
                host_id = "{:02d}".format(host_id)
            except:
                pass
            hostname = prefix[0] + host_id
        curr_time += hostname
    return curr_time


def bug():
    import pdb
    pdb.set_trace()
    # you can use "c" for continue, "p variable", "p locals", "n" for next
    # you can use "!a += 1" for changes of variables
    # you can use "import code; code.interact(local=locals)" to iPython with
    # all variables
if __name__ == "__main__":
    t = show_time(cat_server=True)
    print(t)
