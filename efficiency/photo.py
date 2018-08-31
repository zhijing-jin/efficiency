# -*- coding: utf-8 -*-
from __future__ import division, print_function
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
    d = d.replace(year=this_year) if d.year == 1900 else d.replace(year=d.year)
    return d.strftime(new_format)


def parse_filename(filename):
    root, suffix = filename.rsplit('.', 1)
    suffix = suffix.lower()
    assert suffix in ['jpg', 'png', 'mov', 'mp4', 'gif'], filename + " is not a photo/video file!"
    root = root.rsplit(' (')[0]

    old_format = get_format(root)
    new_format = '%Y%m%d_%H%M%S'

    new_root = transform(root, old_format, new_format)

    return new_root + '.' + suffix


def rename(folders):

    folders = [
                

                '/Users/maggie0/Downloads/20180201_KidsChurch',
'/Users/maggie0/Downloads/20180207_Yilun',
'/Users/maggie0/Downloads/20180601_Kavish',
'/Users/maggie0/Downloads/20180601_Lab',
'/Users/maggie0/Downloads/20180601_Summer',
'/Users/maggie0/Downloads/20180615_JinDiBirthday',
'/Users/maggie0/Downloads/20180616_BrokenPhone',
'/Users/maggie0/Downloads/20180624_HarvardNaturalHistory',
'/Users/maggie0/Downloads/20180629_DanceParty',
'/Users/maggie0/Downloads/20180701_Pika',
'/Users/maggie0/Downloads/20180704_Fireworks',
'/Users/maggie0/Downloads/20180707_SeattleWithYilun',
'/Users/maggie0/Downloads/20180715_KTV',
'/Users/maggie0/Downloads/20180716_Kayak',
'/Users/maggie0/Downloads/20180718_WenbingBirthday',
'/Users/maggie0/Downloads/20180721_Tanglewood',
'/Users/maggie0/Downloads/20180725_Soccer',
'/Users/maggie0/Downloads/20180727_KavishHomeMovie',
'/Users/maggie0/Downloads/20180727_YangAndWinston',
'/Users/maggie0/Downloads/20180728_PartyAtReginas',
'/Users/maggie0/Downloads/20180729_Birthday',
'/Users/maggie0/Downloads/20180731_NightTalkWithJinDi',
'/Users/maggie0/Downloads/20180801_Farewell4West',
'/Users/maggie0/Downloads/20180801_FarewellLab',
'/Users/maggie0/Downloads/20180801_FarewellMIT',
'/Users/maggie0/Downloads/20180803_LaiKingCourt',
'/Users/maggie0/Downloads/20180803_MorningHK',
'/Users/maggie0/Downloads/20180805_TaiMoHiking',
'/Users/maggie0/Downloads/20180810_RegDayHKU',
'/Users/maggie0/Downloads/20180828_Jiang']
    
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
