import multiprocessing
import os


def shell(cmd, working_directory='.', stdout=False, stderr=False, verbose=True):
    import subprocess
    from subprocess import PIPE, Popen

    if verbose:
        print("[Info] Starting to run this command now:", cmd)

    subp = Popen(cmd, shell=True, stdout=PIPE,
                 stderr=subprocess.STDOUT, cwd=working_directory)
    subp_stdout, subp_stderr = subp.communicate()

    if subp_stdout: subp_stdout = subp_stdout.decode("utf-8")
    if subp_stderr: subp_stderr = subp_stderr.decode("utf-8")

    if stdout and subp_stdout:
        print("[stdout]", subp_stdout, "[end]")
    if stderr and subp_stderr:
        print("[stderr]", subp_stderr, "[end]")

    return subp_stdout, subp_stderr



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


def mproc_with_shared_list(func, input_list,
                           avail_cpu=multiprocessing.cpu_count() - 4):
    from multiprocessing import Manager
    from multiprocessing import Pool

    with Manager() as mgr:
        output = mgr.list([])

        # build list of parameters to send to starmap
        params = [[output, param] for param in input_list]

        with Pool(processes=min(len(input_list), avail_cpu)) as p:
            p.starmap(func, params)
        print(output)
        another_list = output[:]
    return another_list


def random_sample(data, size=1000, return_list=True):
    import random
    if_dict = isinstance(data, dict)
    if if_dict:
        data = list(data.items())
    if isinstance(data, set):
        data = list(data)
    random.shuffle(data)
    sample = data[:size]
    if (not return_list) and if_dict:
        sample = dict(sample)
    return sample


def flatten_list(nested_list):
    from itertools import chain
    return list(chain.from_iterable(nested_list))


def flatten_dict(dict_list):
    from collections import ChainMap
    return dict(ChainMap(*dict_list))

def nested_list2tuple(t):
    return tuple(map(nested_list2tuple, t)) if isinstance(t, (tuple, list)) else t

def lstrip_word(word, pref):
    if word.startswith(pref):
        return word[len(pref):]
    return word


def rstrip_word(word, suf):
    if word.endswith(suf):
        return word[:-len(suf)]
    return word


def set_seed(seed=0, verbose=False):
    import random
    import os

    if seed is None:
        from efficiency.log import show_time
        seed = int(show_time())
    if verbose: print("[Info] seed set to: {}".format(seed))

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def dict_diff(dict0, dict1, note=''):
    if isinstance(dict0, dict) and isinstance(dict1, dict):
        for [key0, val0], [key1, val1] in zip(sorted(dict0.items()),
                                              sorted(dict1.items())):
            if key0 != key1:
                yield (key0, key1, note + ', key_is_diff')
            else:
                if isinstance(val0, dict) and isinstance(val1, dict):
                    for i in dict_diff(val0, val1, note + ', ' + key0):
                        for b in i:
                            yield b
                elif (isinstance(val0, list) or isinstance(val0, tuple)) and \
                        (isinstance(val1, list) or isinstance(val1, tuple)):
                    for i in dict_diff(val0, val1, note + ', ' + key0):
                        for b in i:
                            yield b
                else:
                    if val0 != val1:
                        yield (
                        val0, val1, note + ', {}, val_is_diff'.format(key0))

    elif (isinstance(dict0, list) or isinstance(dict0, tuple)) and \
            (isinstance(dict1, list) or isinstance(dict1, tuple)):
        for ls_ix, (item0, item1) in enumerate(zip(dict0, dict1)):
            for i in dict_diff(item0, item1, note + ', {}'.format(ls_ix)):
                for b in i:
                    yield b
    else:
        if dict0 != dict1:
            yield (dict0, dict1, note)

def reorder(_x, order):
    x = list(range(len(_x)))
    for i, a in zip(order, _x):
        x[i] = a
    return x


def avg(num_list, decimal=2, return_std=False):
    if not len(num_list):
        if return_std:
            return 0, 0
        return 0

    import numpy as np
    mean = np.nanmean(num_list)
    mean = round(mean, decimal)

    if return_std:
        std = np.std(num_list)
        std = round(std, decimal)
        return mean, std
    return mean


def if_same_len(num_list):
    return len(set(num_list)) == 1


def load_yaml(yaml_filepath, dir_=None, op=lambda x: x):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    import yaml

    def make_paths_absolute(dir_, cfg):
        """
        Make all values for keys ending with `_path` absolute to dir_.

        Parameters
        ----------
        dir_ : str
        cfg : dict

        Returns
        -------
        cfg : dict
        """
        for key in cfg.keys():
            if key.endswith("_path") or key.endswith("_dir"):
                cfg[key] = os.path.join(dir_, cfg[key])
                cfg[key] = os.path.abspath(cfg[key])
                if not os.path.exists(cfg[key]):
                    print("[Warn] %s does not exist.", cfg[key])
            if type(cfg[key]) is dict:
                cfg[key] = make_paths_absolute(dir_, cfg[key])
        return cfg

    if dir_ is None:
        dir_ = os.path.dirname(yaml_filepath)
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.safe_load(stream)

    cfg = op(cfg)

    cfg = make_paths_absolute(dir_, cfg)

    return cfg


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    set_seed(0)
    print(if_same_len([3, 23, 3]))
    print(if_same_len([3, 3, 3]))
