import multiprocessing
import os


def shell(cmd, working_directory='.', stdout=False, stderr=False):
    import sys
    import os
    import subprocess
    from subprocess import PIPE, Popen

    subp = Popen(cmd, shell=True, stdout=PIPE,
                 stderr=subprocess.STDOUT, cwd=working_directory)
    subp_stdout, subp_stderr = subp.communicate()

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


def set_seed(seed=0):

    import random

    if seed is None:
        from efficiency.log import show_time
        seed = int(show_time())
    print("[Info] seed set to: {}".format(seed))

    random.seed(seed)
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
    except ImportError:
        pass


def reorder(_x, order):
    x = list(range(len(_x)))
    for i, a in zip(order, _x):
        x[i] = a
    return x


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
