from os.path import isdir, dirname, realpath, basename
from os import listdir, path
from efficiency.function import shell
from efficiency.log import show_var


class FileManager():

    def __init__(self, dir=dirname(realpath(__file__)), file_filter=lambda f: True):
        self.files = self.recurse_files(dir, file_filter=file_filter)

    @staticmethod
    def recurse_files(folder, file_filter=lambda f: True):
        if isdir(folder):
            return [path.join(folder, f) for f in listdir(folder)
                    if file_filter(f)]
        return [folder]

    def rename_files(self, prefix='my_MP3_'):
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
    fm.rename_files(prefix='my_MP3_')
