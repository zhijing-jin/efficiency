import multiprocessing


def reorder(_x, order):
    x = range(len(_x))
    for i, a in zip(order, _x):
        x[i] = a
    return x


def do_nothing(a):
    return a


def shell(cmd, show_res=False):
    import sys
    import os
    import subprocess
    from subprocess import PIPE, Popen

    subp = Popen(cmd, shell=True, stdout=PIPE, stderr=subprocess.STDOUT)
    subp_output = subp.communicate()[0]

    if show_res:
        print("Here is the output:", subp_output, "[[end]]")
    return subp_output


def mproc(func, input_list, avail_cpu=multiprocessing.cpu_count() - 4):
    '''
    This is a multiprocess function where you execute the function with 
    every input in the input_list simutaneously.
    @ return output_list: the list of outputs w.r.t. input_list
    '''
    from multiprocessing import Pool
    pool = Pool(processes=min(len(input_list), avail_cpu))
    output_list = pool.map(func, input_list)
    return output_list
