class Constant:
    file_acl_anthology = 'anthology.bib'
    file_acl_anthology_subset_format = 'custom_anthology_{}.bib'
    file_temp_bib_in = 'temp_in.bib'
    file_rename_in = 'temp_in.bib'
    file_rename_out = 'temp_out.bib'

    proper_nouns = {'ai', 'nlp', 'eu', 'bert', 'roberta', 'electra', 'lstm', 'cnn', 'rnn', 'wmt', 'nmt', 'europarl', }
    proper_nouns |= {'Kolmogorov', 'MDL', 'Markov'}
    proper_nouns = {i.lower() for i in proper_nouns}

    def __init__(self):
        import re
        self.str2clean = lambda i: re.sub("[^a-zA-Z ]+", "", i.lower())
        self.en_stop_words = self._get_stop_words()['en']

    @staticmethod
    def _get_stop_words():
        from nltk.corpus import stopwords

        lang_abbrev_to_fullname = {'en': 'english', 'fr': 'french', 'de': 'german', 'es': 'spanish'}

        lang2stop_words = {}
        for abbrev, fullname in lang_abbrev_to_fullname.items():
            stop_words = stopwords.words(fullname)
            if fullname == 'english':
                stop_words.extend(['european', 'union', 'eu'])
            elif fullname == 'french':
                stop_words.extend(['européens', 'union', 'eu'])
            elif fullname == 'german':
                stop_words.extend(['europäische', 'union', 'eu'])
            elif fullname == 'spanish':
                stop_words.extend(['Europea', 'unión', 'eu'])
            lang2stop_words[abbrev] = set(stop_words)
        return lang2stop_words


class BibCleaner:
    def __init__(self, file_in):
        try:
            from pybtex.database.input import bibtex
        except:
            import os
            os.system('pip install pybtex')

        from pybtex.database.input import bibtex

        parser = bibtex.Parser()
        from efficiency.log import show_var, show_time
        show_time(f'Start reading {file_in}')

        self.bib_data = parser.parse_file(file_in)

        parser = bibtex.Parser()
        self.bib_out = parser.parse_file(C.file_temp_bib_in)
        show_time(f'Finished reading {len(self.bib_data.entries)} bib entries from {file_in}')

    def extract_subset(self, file_out_format, batch_size=1800):
        bib_data = self.bib_data
        papers_needed = self._old_papers2new_papers_by_keyword(bib_data.entries.items())
        papers_needed = self._old_papers2new_papers_with_new_key(papers_needed)

        from efficiency.log import show_var, show_time
        show_var(['len(papers_needed)'])

        import numpy as np
        num_batches = int(np.ceil(len(papers_needed) / batch_size))
        for batch_ix in range(num_batches):
            papers = papers_needed[batch_ix * batch_size: (batch_ix + 1) * batch_size]
            file = file_out_format.format(batch_ix)
            self._paper_list2file(papers, file)

    def rename_key(self, file_out):
        bib_data = self.bib_data
        papers_needed = self._old_papers2new_papers_with_new_key(bib_data.entries.items())
        from efficiency.log import show_var, show_time
        show_var(['len(papers_needed)'])

        self._paper_list2file(papers_needed, file_out)

    def _paper_list2file(self, list_of_key_n_paper, file):
        from pybtex.utils import OrderedCaseInsensitiveDict
        self.bib_out.entries = OrderedCaseInsensitiveDict(list_of_key_n_paper)

        self.bib_out.to_file(file)
        print(f'[Info] Saved {len(list_of_key_n_paper)} bib entries to {file}')

    @staticmethod
    def _old_papers2new_papers_by_keyword(old_papers):
        from tqdm import tqdm
        papers_needed = []
        for key, paper in tqdm(old_papers):
            title = paper.fields['title']
            title_toks = set(title.split())

            if ('translat' in title.lower()) or ('MT' in title_toks):
                papers_needed.append((key, paper))
            else:
                pass
                # del bib_data.entries[key] # Not Implemented
        return papers_needed

    def _old_papers2new_papers_with_new_key(self, old_papers):
        from tqdm import tqdm

        papers_needed = []
        old_paper_keys = {key for key, _ in old_papers}

        import re
        for key, paper in tqdm(old_papers):
            print('Working on', key)

            # Function 1: Rename the key
            if any(key.startswith(i) for i in ['JMLR:', 'DBLP:', 'doi:']) \
                    or any(c in key for c in '/:'):
                title = C.str2clean(paper.fields['title'].lower()).split()
                title = [i for i in title if i not in C.en_stop_words]
                year = paper.fields['year']

                last_name = paper.persons['author'][0].last_names[0].lower()
                last_name = re.sub(r'[^a-zA-Z]', '', last_name)  # in case of Demner{-}Fushman, M{\"{o}}kander

                key = f'{last_name}{year}{title[0]}'
                print('Renamed to', key)
                if key in old_paper_keys:
                    key += title[1]
                    if key in old_papers:
                        import pdb;
                        pdb.set_trace()

            # Function 1.5: Remove the abstract
            fields_to_delete = ['abstract']  # because this is usually lengthy and sometimes contains error characters
            for field in fields_to_delete:
                if field in paper.fields:
                    del (paper.fields['abstract'])

            # Function 2: Update capitalization in the title
            raw_title = paper.fields['title']
            raw_title_toks = raw_title.split()
            new_title_toks = ['{' + i + '}' if i.lower() in C.proper_nouns else i for i in raw_title_toks]
            new_title = ' '.join(new_title_toks)

            new_title = new_title.replace('BERT-', '{BERT}-')

            for punct in [':', '?']:
                punct_occurrence = punct + ' '
                if punct_occurrence in new_title:
                    colon_pos = new_title.index(punct_occurrence)
                    insert_pos = colon_pos + 2
                    if new_title[insert_pos] != '{':
                        new_title = new_title[:insert_pos] + '{' + new_title[insert_pos].upper() + '}' + new_title[
                                                                                                         insert_pos + 1:]
            paper.fields['title'] = new_title

            papers_needed.append((key, paper))
            print('Finished:', key)
        return papers_needed


def clean_bib_file(input_file='temp_in.bib', output_file='temp_out.bib'):
    bc = BibCleaner(input_file)
    bc.rename_key(output_file)


if __name__ == '__main__':
    C = Constant()

    bc = BibCleaner(C.file_rename_in, )
    bc.rename_key(C.file_rename_out)

    # bc = BibCleaner(C.file_acl_anthology)
    # bc.extract_subset(C.file_acl_anthology_subset_format)
