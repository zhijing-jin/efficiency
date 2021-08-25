from __future__ import unicode_literals, print_function
from efficiency.log import show_var


class NLP:
    def __init__(self):
        import spacy

        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'tagger', "lemmatizer"])
        try:
            self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        except:
            self.nlp.add_pipe('sentencizer')

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)
    
    @staticmethod
    def sent_bleu(ref_list, hyp):
        from nltk.translate import bleu
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method4
        refs = [ref.split() for ref in ref_list]
        hyp = hyp.split()
        return bleu(refs, hyp, smoothing_function=smoothie)

def main():
    raw_text = 'Hello, world. Here are two people with M.A. degrees from UT Austin. This is Mr. Mike.'
    nlp = NLP()
    sentences = nlp.sent_tokenize(raw_text)
    words = nlp.word_tokenize(sentences[0], lower=True)
    show_var(['sentences', 'words'])


if __name__ == '__main__':
    main()
