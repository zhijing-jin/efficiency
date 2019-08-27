from os.path import isdir, dirname, realpath, basename
from os import listdir, path
from itertools import chain
from efficiency.function import shell
from efficiency.log import show_var


class FileManager():

    def __init__(self, ext='.mp3'):
        self.files = self.recurse_files(dirname(realpath(__file__)))

    def recurse_files(self, folder, filter=lambda f: True):
        if isdir(folder):
            return [path.join(folder, f) for f in listdir(folder)
                    if filter(f)]
        return [folder]

    def rename_files(self, prefix='My_mp3_'):
        for f in self.files:
            dir = dirname(f)
            fname = basename(f)
            new_fname = prefix + fname
            new_f = path.join(dir, new_fname)
            cmd = 'mv "{f}" "{new_f}"'.format(f=f, new_f=new_f)
            show_var(['cmd'])
            shell(cmd)


if __name__ == '__main__':
    fm = FileManager()
    fm.rename_files(prefix='My_mp3_')
