from datetime import datetime
import pdb
from function import shell
from os import listdir
from os.path import isfile, join
from log import *


def get_format(text):
    format_list = ['Photo %d-%m-%Y, %H %M %S',
                   'Video %d-%m-%Y, %H %M %S',
                   'Photo %b %d, %I %M %S %p',
                   'Video %b %d, %I %M %S %p']
    right_form = None
    for form in format_list:
        try:
            datetime.strptime(text, form)
        except ValueError:
            continue
        right_form = form
        break
    if right_form == None:
        assert False, "No matched format for " + text
    return right_form


def transform(text, old_format, new_format, this_year=datetime.today().year):
    d = datetime.strptime(text, old_format)
    d = d.replace(year=this_year) if d.year == 1900 else d.year
    return d.strftime(new_format)


def parse_filename(filename):
    root, suffix = filename.rsplit('.', 1)
    suffix = suffix.lower()
    assert suffix in ['jpg', 'png', 'mov'], filename + " is not a photo/video file!"
    root = root.rsplit(' (')[0]

    old_format = get_format(root)
    new_format = '%Y%m%d_%H%M%S'

    new_root = transform(root, old_format, new_format)

    return new_root + '.' + suffix


def rename(folders):

    folders = ['/Users/Fascinating/Dropbox (MIT)/LifeTrace/image/temp']
    for folder in folders:
        files = [f for f in listdir(folder)
                 if isfile(join(folder, f)) and
                 (not f.startswith('.')) and
                 (f.startswith('Photo ') or f.startswith('Video '))]
        pdb.set_trace()
        for filename in files:
            new_name = parse_filename(filename)
            cmd = "mv '{}' {}".format(filename, new_name)
            show_var(["cmd"])
            pdb.set_trace()
            shell(cmd, working_directory=folder, stdout=True, stderr=True)


if __name__ == "__main__":
    filename = 'Photo 1-1-2017, 12 03 16 (1).jpg'
    'Video 25-12-2017, 05 22 22.mov'
    'Photo Jun 24, 10 11 11 PM'
    'Photo Jun 24, 5 14 02 AM.jpg'
    'IMG_1145.JPG'
    'WechatIMG123.jpeg'

    rename('')
