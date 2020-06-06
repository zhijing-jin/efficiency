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


def torchload(path, verbose=True, timetaking=True):
    import torch
    if verbose and timetaking:
        show_time('[Info] Loading from {}'.format(path))
    data = torch.load(path)
    if verbose:
        printout = '[info] Loaded {} object from {}'.format(len(data), path)
        if timetaking:
            show_time(printout)
        else:
            print(printout)
    return data


def torchsave(dic, path, verbose=True, timetaking=True, check_exist=False):
    import os
    import torch
    if check_exist and os.path.isfile(path):
        print('[Warn] tempering', path)
        import pdb
        pdb.set_trace()
    if verbose:
        printout = '[info] Saving {} object to {}'.format(len(dic), path)
        if timetaking:
            show_time(printout)
        else:
            print(printout)
    torch.save(dic, path)
    if timetaking:
        printout = '[info] Saved {} object to {}'.format(len(dic), path)
        show_time(printout)


def write_var(var, path='data/debug/var'):
    with open(path, 'w') as f:
        f.write(path.split('/')[-1] +
                ' = ' + repr(var) + '\n')


def smart_json_dumps(data_structure, file_path='', make_lists_no_indent=True):
    import re
    import json

    class _NoIndent(object):
        """ Value wrapper. """

        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return repr(self.value)

    class _NoIndentEncoder(json.JSONEncoder):
        FORMAT_SPEC = '@@{}@@'
        regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

        def __init__(self, **kwargs):
            # Save copy of any keyword argument values needed for use here.
            self.__sort_keys = kwargs.get('sort_keys', None)
            super().__init__(**kwargs)

        def default(self, obj):
            return (
                self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, _NoIndent)
                else super().default(obj))

        def encode(self, obj):
            from _ctypes import PyObj_FromPtr

            format_spec = self.FORMAT_SPEC  # Local var to expedite access.
            json_repr = super().encode(obj)  # Default JSON.

            # Replace any marked-up object ids in the JSON repr with the
            # value returned from the json.dumps() of the corresponding
            # wrapped Python object.
            for match in self.regex.finditer(json_repr):
                # see https://stackoverflow.com/a/15012814/355230
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_obj_repr = json.dumps(no_indent.value,
                                           sort_keys=self.__sort_keys)

                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                json_repr = json_repr.replace(
                    '"{}"'.format(format_spec.format(id)), json_obj_repr)

            return json_repr

    def _make_all_lists_no_indent(data):
        if isinstance(data, dict):
            return {k: _make_all_lists_no_indent(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return _NoIndent(data)
        else:
            return data

    def _key_triple2str(data):
        if isinstance(data, dict):
            return {str(k) if isinstance(k, tuple) else k
                    : v for k, v in data.items()}
        else:
            return data

    new_data = _key_triple2str(data_structure)
    if make_lists_no_indent:
        new_data = _make_all_lists_no_indent(new_data)

    text = json.dumps(new_data, cls=_NoIndentEncoder, sort_keys=True, indent=2)
    if file_path:
        (text, file_path)
    return text


def fwrite(new_doc, path, mode='w', no_overwrite=False, verbose=False):
    import os
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    if no_overwrite and os.path.isfile(path):
        print("[Error] pls choose whether to continue, as file already exists:",
              path)
        import pdb
        pdb.set_trace()
        return
    if verbose:
        try:
            import ast
            data = ast.literal_eval(new_doc)
            if isinstance(data, dict) or isinstance(data, list):
                length = len(data)
                print('[Info] Writing {} samples into {}'.format(length, path))
        except:
            length = new_doc.count('\n') + 1
            print('[Info] Writing {} lines into {}'.format(length, path))

    with open(path, mode) as f:
        f.write(new_doc)


def fread(path, if_strip=False):
    with open(path, 'r') as f:
        data = f.readlines()
    if if_strip:
        data = [line.strip() for line in data]
    return data


def show_time(what_happens='', cat_server=False, printout=True):
    import datetime

    disp = '⏰ Time: ' + \
           datetime.datetime.now().strftime('%m%d%H%M-%S')
    disp = disp + '\t' + what_happens if what_happens else disp
    if printout:
        try:
            print(disp)
        except:
            pass
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


def get_git_version():
    import os
    try:
        import git
    except ImportError:
        os.system('pip install --user gitpython')
        import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    return sha


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
