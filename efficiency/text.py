# -*- coding: utf-8 -*-
from __future__ import division, print_function
import pickle
import random
import numpy as np
import json
import sys
import os.path
import argparse
import os

import pprint
import pdb
from efficiency.log import fwrite, show_var

from bs4 import BeautifulSoup
from collections import defaultdict, Counter, OrderedDict


def format_text(format, text, attr=''):
    if text:
        if attr:
            attr = ' ' + attr
        return "<{f}{}>{t}</{f}>".format(f=format, t=text, a=attr)


def soup_tag(soup, text, tag_symbol, **kwargs):
    # import pdb
    # pdb.set_trace()
    tag = soup.new_tag(tag_symbol, **kwargs)
    if text is not None:
        tag.string = text

    return tag


def example():
    soup = BeautifulSoup(features="html.parser")
    ls = []
    ls += ["Hello world "]

    tag = soup_tag(soup, "red", 'font', style="border:2px solid Tomato;")
    ls += [tag]

    tag = soup_tag(soup, "maggie", "sub")
    ls += [tag]

    tag = soup_tag(soup, "important things", "b")
    ls += [tag]

    ls += [soup_tag(soup, None, "br")]
    ls += [" are transient."]
    ls += [soup.new_tag("br")]
    ls += ["end."]

    soup = soupls2html(soup, ls)

    show_var(["soup"])  # , "soup.prettify()"

    return soup


class HtmlSymb(object):

    def __init__(self):
        self.sub = 'sub'
        self.sup = 'sup'
        self.italic = 'i'

        self.bg_color = ["snow", "honeydew", "mintcream", "azure", "aliceblue", "ghostwhite", "whitesmoke", "seashell",
                         "beige", "oldlace", "floralwhite", "ivory", "antiquewhite", "linen", "lavenderblush", "mistyrose"]
        self.bg_color = ["#d6cbd3", "#eca1a6", "#bdcebe",
                         "#d5e1df", "#e3eaa7", "#b5e7a0", "#86af49"]


def soupls2html(soup, soup_ls):

    for i in soup_ls:
        soup.append(i)
    soup = "<!DOCTYPE html><html><body>{}</body></html>".format(repr(soup))
    return soup


def stylize_text(text_pattern, conll_file, tag_ls,
                 soup=BeautifulSoup(features="html.parser"), html_sym=HtmlSymb()):
    assert isinstance(
        text_pattern, defaultdict), "input is not defaultdict Type."
    word = text_pattern['word']
    gold = text_pattern['gold']
    pred = text_pattern['pred']
    word_n = text_pattern['word_n']
    gold_n = text_pattern['gold_n']
    pred_n = text_pattern['pred_n']

    next_line_exists = text_pattern['next_line_exists']

    soup_ls = []

    if word in ['\n', '</s>']:
        soup_ls += [soup_tag(soup, None, "br")]
    elif conll_file.endswith('.c_w_d_dw_ds_sw_word_ibo_dic'):
        for tag_i, tag_type in enumerate(tag_ls):
            if tag_type in gold:
                word_text = "[{}]".format(word)
                style = "background-color:{};".format(html_sym.bg_color[tag_i])
                tag = soup_tag(soup, word_text, "b", style=style)
                style = "color:{};".format(html_sym.bg_color[tag_i])
                sub_tag = soup_tag(
                    soup, tag_type, "sub", style=style)
                soup_ls += [tag, sub_tag]
                break
        if not soup_ls:
            soup_ls += [word]

    elif conll_file.endswith('.conll'):
        if (gold == pred) and (gold == "O"):

            soup_ls += [word]
        elif (gold == pred):
            tag = soup_tag(soup, word, "b")
            soup_ls += [tag]
        elif (gold != pred):
            word_text = "[{}]".format(word)
            if_end_word = (not next_line_exists) or \
                ((gold != "O") and (gold_n == "O"))
            sub_text = gold
            if if_end_word:
                sub_text += ")"
            tag_after_sub = soup_tag(
                soup, sub_text, "sub", style="color:DodgerBlue;")

            tag = soup_tag(soup, word_text, "b",
                           style="background-color:Orange;")

            if_end_word = (not next_line_exists) or \
                ((pred != "O") and (pred_n == "O"))
            sup_text = pred
            if if_end_word:
                sup_text += ")"
            tag_after_sup = soup_tag(
                soup, sup_text, "sup", style="color:Tomato;")

            this_word_format = [tag, tag_after_sub, tag_after_sup]

            if (gold.startswith("B-")):
                tag_before_sub = soup_tag(
                    soup, "(", "sub", style="color:DodgerBlue;")
                this_word_format = [
                    tag_before_sub] + this_word_format
            if (pred.startswith("B-")):
                tag_before_sup = soup_tag(
                    soup, "(", "sup", style="color:Tomato;")
                this_word_format = [
                    tag_before_sup] + this_word_format
            soup_ls += this_word_format
    return soup_ls


class ConllSymb(object):

    def __init__(self):
        self.EOS = '</s>'
        self.SOD = '-DOCSTART-'


def conll_info(contents, field_w, field_g, conll_symb=ConllSymb()):

    words = [line.split('\t')[field_w] for line in contents if line.strip()]
    tags = [line.split('\t')[field_g] for line in contents if line.strip()]

    tag_cnt = Counter(tags)
    tag_cnt = OrderedDict(sorted(list(tag_cnt.items()),
                                 key=lambda i: i[0][2:] + i[0][:2]))
    tag_ls = sorted(list(set(i[2:] for i in tag_cnt.keys() if i != "O")))

    dataset_n_tags = [num for t, num in tag_cnt.items() if t != "O"]
    dataset_n_tags = sum(dataset_n_tags)

    dataset_n_docs = words.count(conll_symb.SOD)
    dataset_n_sents = words.count(conll_symb.EOS)
    dataset_n_words = len(words) - dataset_n_sents

    dataset_tag_dens = dataset_n_tags / dataset_n_words
    sent_tag_dens = dataset_n_tags / dataset_n_sents

    EOS_ixs = [i for i, w in enumerate(words) if w == conll_symb.EOS]
    sent_n_words = [end - st for st, end in zip([0] + EOS_ixs, EOS_ixs)]
    sent_n_words_avg = sum(sent_n_words) / len(sent_n_words)

    SOD_ixs = [i for i, w in enumerate(words) if w == conll_symb.SOD]
    docs = [words[st:end] for st, end in zip(SOD_ixs, SOD_ixs[1:] + [None])]

    doc_n_words = [len(d) for d in docs]
    doc_n_words_avg = sum(doc_n_words) / len(doc_n_words)

    doc_n_sents = [d.count(conll_symb.EOS) for d in docs]
    doc_n_sents_avg = sum(doc_n_sents) / len(doc_n_sents)

    word_n_chars = [len(w) for w in words]
    word_n_chars_avg = sum(word_n_chars) / len(word_n_chars)

    info = OrderedDict([("tag_ls", tag_ls),
                        ("tag_cnt", tag_cnt),

                        ("dataset_n_tags", "{} tags".format(
                            dataset_n_tags)),
                        ("dataset_n_words", "{} words".format(
                            dataset_n_words)),
                        ("dataset_n_sents", "{} sents".format(
                            dataset_n_sents)),
                        ("dataset_n_docs", "{} docs".format(
                            dataset_n_docs)),

                        ("dataset_tag_dens",
                         "{:.2f} tag/word".format(dataset_tag_dens)),
                        ("sent_tag_dens",
                         "{:.2f} tag/sentenece".format(sent_tag_dens)),


                        ("doc_n_words_avg",
                         "{:.2f} word/document".format(doc_n_words_avg)),
                        ("doc_n_sents_avg",
                         "{:.2f} sent/document".format(doc_n_sents_avg)),

                        ("sent_n_words_avg",
                         "{:.2f} word/sent".format(sent_n_words_avg)),
                        ("word_n_chars_avg",
                         "{:.2f} char/word".format(word_n_chars_avg))
                        ])

    return info


def conll2html(conll_file, html_sym=HtmlSymb(), conll_symb=ConllSymb(),
               soup=BeautifulSoup(features="html.parser"), field_w=-3, field_g=-2, field_p=-1):
    with open(conll_file) as f:
        if conll_file.endswith(".c_w_d_dw_ds_sw_word_ibo_dic"):
            fields = f.readline().strip()
            fields = fields.split('\t')
            field_w = fields.index('word')
            field_g = fields.index('tag')
            field_p = None
        raw_conll = f.readlines()

    soup_ls = []

    if conll_file.endswith(".c_w_d_dw_ds_sw_word_ibo_dic"):
        meta_dic = conll_info(raw_conll, field_w, field_g)
        import json2table
        meta_html = json2table.convert(
            meta_dic, table_attributes={"style": "width:50%", "class": "table table-striped", "border": 1})
        soup_ls += [BeautifulSoup(meta_html, features="html.parser")]

        soup_ls += [soup_tag(soup, None, "br")] * 10

        tag_ls = meta_dic['tag_ls']
        for tag_i, tag_type in enumerate(tag_ls):
            style = "background-color:{};".format(html_sym.bg_color[tag_i])
            tag = soup_tag(soup, tag_type, "b", style=style)
            soup_ls += [tag, '    ']

        soup_ls += [soup_tag(soup, None, "br")] * 10
    else:
        tag_ls = ['LOC', 'MISC', 'ORG', 'PER']
    for line_i, line in enumerate(raw_conll):
        word_pattern = defaultdict(str)

        # check next_line_exists
        next_line_exists = ((line_i + 1) < len(raw_conll))
        if next_line_exists:
            line_n = raw_conll[line_i + 1]
            if (line_n == '\n') or (conll_symb.EOS in line_n):
                next_line_exists = False
        word_pattern.update({"next_line_exists": next_line_exists})

        if line == '\n':
            word_pattern.update({"word": line})
        else:
            tokens = line.split()
            word = tokens[field_w]
            gold = tokens[field_g]
            word_pattern.update({"word": word,
                                 "gold": gold})
            if conll_file.endswith('.conll'):
                pred = tokens[field_p]
                word_pattern.update({"pred": pred})

            if next_line_exists:
                line_n = raw_conll[line_i + 1]
                tokens_n = line_n.split()

                word_n = tokens_n[field_w]
                gold_n = tokens_n[field_g]
                word_pattern.update({"word_n": word_n,
                                     "gold_n": gold_n})
                if conll_file.endswith('.conll'):
                    pred_n = tokens_n[field_p]
                    word_pattern.update({"pred_n": pred_n})

        soup_ls += stylize_text(word_pattern, conll_file, tag_ls)

        if line != '\n':
            soup_ls += [' ']

    return soupls2html(soup, soup_ls)


def get_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='train.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--conll', type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    raw_n_html = [
        ('data/03conll.train.c_w_d_dw_ds_sw_word_ibo_dic', 'demo/03conll.train.html'),
        ('data/03conll.valid.c_w_d_dw_ds_sw_word_ibo_dic', 'demo/03conll.valid.html'),
        ('data/sample.c_w_d_dw_ds_sw_word_ibo_dic', 'dana.sample.html'),
        ('data/lstm_encoder_errors.conll', 'demo/lstm_encoder_errors.html'),
        ('data/cnn_encoder_errors.conll', 'demo/cnn_encoder_errors.html'),
        ('data/common_sents_diff_error.conll',
         'demo/common_sents_diff_error.html'),
        ('data/common_sents_same_error.conll', 'demo/common_sents_same_error.html')




    ]
    args = get_args()
    if args.conll:
        raw_n_html = [('data/{}.conll'.format(args.conll),
                       'demo/{}.html'.format(args.conll))]
    for conll_file, html_file in raw_n_html:
        print("[Info] Visualizing {} into {}".format(conll_file, html_file))
        soup = conll2html(conll_file)
        fwrite(soup, html_file)

    # soup = example()
