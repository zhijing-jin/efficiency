  # -*- coding: utf-8 -*-
from __future__ import division, print_function

def get_html_n_save(url, file=None):
    import os
    if file is not None:
        if os.path.isfile(file):
            html = open(file).read()
            return html

    import requests
    r = requests.get(url)

    if r.status_code == 200:
        from efficiency.log import fwrite

        html = r.text
        if file is not None:
            fwrite(html, file, verbose=True)
        return html
    else:
        print('[Error] {} for {}'.format(r.status_code, url))
        return None
