# -*- coding: utf-8 -*-
from __future__ import division, print_function
import random
import sys
import os.path
import socket
import pdb


def show_var(expression,
             joiner='\n', print=print):
    '''
    Prints out the name and value of variables. 
    Eg. if a variable with name `num` and value `1`,
    it will print "num: 1\n"

    Parameters
    ----------
    expression: ``List[str]``, required
        A list of varible names string.

    Returns
    ----------
        None
    '''

    import json

    var_output = []

    for var_str in expression:
        frame = sys._getframe(1)
        value = eval(var_str, frame.f_globals, frame.f_locals)

        if ' object at ' in repr(value):
            value = vars(value)
            value = json.dumps(value, indent=2)
            var_output += ['{}: {}'.format(var_str, value)]
        else:
            var_output += ['{}: {}'.format(var_str, repr(value))]

    if joiner != '\n':
        output = "[Info] {}".format(joiner.join(var_output))
    else:
        output = joiner.join(var_output)
    print(output)
    return output


def torchsave(dic, path):
    import torch
    if os.path.isfile(path):
        print('[Warn] tempering', path)
        import pdb
        pdb.set_trace()
    print('[info] saving object to', path)
    torch.save(dic, path)


def write_var(var, path='data/debug/var'):
    with open(path, 'w') as f:
        f.write(path.split('/')[-1] +
                ' = ' + repr(var) + '\n')


def fwrite(new_doc, path, mode='w', no_overwrite=False):
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    if no_overwrite and os.path.isfile(path):
        print("[Error] pls choose whether to continue, as file already exists:", path)
        import pdb
        pdb.set_trace()
        return
    with open(path, mode) as f:
        f.write(new_doc)


def fread(path):
    with open(path, 'r') as f:
        return f.readlines()


def show_time(what_happens='', cat_server=False, printout=True):
    import datetime

    disp = '⏰ Time: ' + \
        datetime.datetime.now().strftime('%m%d%H%M-%S')
    disp = disp + '\t' + what_happens if what_happens else disp
    if printout:
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
        else:
            hostname = hostname[0]
        curr_time += hostname
    return curr_time


def del_quote(string):
    cleaned = string.replace("'", "")
    cleaned = cleaned.replace("\"", "")
    return cleaned


def gpu_mem(gpu_id=0):
    from efficiency.function import shell

    line = 9 + gpu_id * 3
    cmd = "nvidia-smi | head -n {} | tail -n 1 | awk '{{print $9}}' | sed 's/MiB//' ".format(
        line)

    stdout, stderr = shell(cmd)

    stdout = stdout.strip()
    mem = int(stdout) if stdout != b'' else None

    return mem


def debug(what_to_debug=''):
    if what_to_debug:
        print("[Info] start debugging {}".format(what_to_debug))

    import pdb
    pdb.set_trace()

    # you can use "c" for continue, "p variable", "p locals", "n" for next
    # you can use "!a += 1" for changes of variables
    # you can use "import code; code.interact(local=locals)" to iPython with
    # all variables
if __name__ == "__main__":

    a = "something"
    b = 1
    show_var(["a", "b"], joiner=', ')

    debug("show_var")

    show_var(["c"])
