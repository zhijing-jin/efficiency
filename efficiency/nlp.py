from __future__ import unicode_literals, print_function
from efficiency.log import show_var


class NLP:
    def __init__(self, disable=['ner', 'parser', 'tagger', "lemmatizer"]):
        import spacy

        self.nlp = spacy.load('en_core_web_sm', disable=disable)
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


class Chatbot:
    model_version2engine = {
        'gpt3': "text-davinci-003",
        'gpt3.5': "gpt-3.5-turbo",
        'gpt4': "gpt-4",
    }
    engine2pricing = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4-32k": 0.12,
        "gpt-4": 0.06,
        "text-davinci-003": 0.0200,
    }

    def __init__(self, model_version='gpt3.5', max_tokens=100, output_file='.cache_gpt_responses.csv',
                 system_prompt="You are a helpful assistant."):
        import os
        openai_api_key = os.environ['OPENAI_API_KEY']

        self.model_version = model_version
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.output_file = output_file
        self.gpt_files = [output_file]

        import openai
        openai.api_key = openai_api_key
        self.openai = openai
        self.num_tokens = []
        self.cache = self.load_cache()
        # self.list_all_models()
        self.clear_dialog_history()

    def clear_dialog_history(self):
        self.dialog_history = [
            {"role": "system", "content": self.system_prompt},
            # {"role": "user", "content": "Who won the world series in 2020?"},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        ]

    def dialog_history_to_str(self):
        dialog_history_text = []
        for turn in self.dialog_history:
            if turn['role'] == 'user':
                prefix = 'Q'
            elif turn['role'] == 'assistant':
                prefix = 'A'
            else:
                continue
            this_text = f"{prefix}: {turn['content']}"
            if prefix == 'A':
                this_text += '\n'
            dialog_history_text.append(this_text)
        dialog_history_text = '\n'.join(dialog_history_text) + '\nA:'
        return dialog_history_text

    def list_all_models(self):
        model_list = self.openai.Model.list()['data']
        model_ids = [x['id'] for x in model_list]
        model_ids.sort()
        print(model_ids)
        import pdb;
        pdb.set_trace()

    @property
    def _total_cost(self):
        return sum(self.num_tokens) // 1000 * self.engine2pricing[self.engine]

    def print_cost(self):
        print(f"[Info] Spent ${self._total_cost} for {sum(self.num_tokens)} tokens.")

    def load_cache(self):
        cache = {}
        from efficiency.log import fread
        for file in self.gpt_files:
            data = fread(file, verbose=False)
            cache.update({i[f'query{q_i}']: i[f'pred{q_i}'] for i in data
                          for q_i in list(range(10)) + ['']
                          if f'query{q_i}' in i})
        return cache

    def query_with_patience(self, *args, **kwargs):
        import openai
        try:
            return self.query(*args, **kwargs)
        except openai.error.InvalidRequestError:
            import pdb;
            pdb.set_trace()
            if len(self.dialog_history) > 10:
                import pdb;
                pdb.set_trace()
            for turn_i, turn in enumerate(self.dialog_history):
                if turn['role'] == 'assistant':
                    turn['content'] = turn['content'][:1000]

        except openai.error.RateLimitError:
            sec = 100
            print(f'[Info] openai.error.RateLimitError. Wait for {sec} seconds')
            self.print_cost()
            '''
            Default rate limits for gpt-4/gpt-4-0314 are 40k TPM and 200 RPM. Default rate limits for gpt-4-32k/gpt-4-32k-0314 are 80k PRM and 400 RPM. 
            https://platform.openai.com/docs/guides/rate-limits/overview
            '''

            import time
            time.sleep(sec)
            return self.query_with_patience(*args, **kwargs)

    def query(self, question, turn_off_cache=False, stop_sign="\nQ: ",
              continued_questions=False, max_tokens=None, verbose=True, only_response=True,
              model_version=[None, 'gpt3', 'gpt3.5', 'gpt4'][0],
              engine=[None, "text-davinci-003", "gpt-3.5-turbo", "gpt-4-32k-0314", "gpt-4-0314", "gpt-4"][0],
              enable_pdb=False,
              ):
        if model_version is not None:
            engine = self.model_version2engine[model_version]
        elif engine is not None:
            engine = engine
        else:
            engine = self.model_version2engine[self.model_version]

        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        verbose = True if enable_pdb else verbose

        self.engine = engine
        if_newer_engine = engine.startswith('gpt-3.5') or engine.startswith('gpt-4')

        if not continued_questions:
            self.clear_dialog_history()

        self.dialog_history.append({"role": "user", "content": question}, )

        if (question in self.cache) & (not turn_off_cache):
            response_text = self.cache[question]
            if not if_newer_engine: response_text = response_text.split(stop_sign, 1)[0]
            if verbose: print(f'[Info] Using cache for {question}')
        else:
            openai = self.openai
            if if_newer_engine:
                response = openai.ChatCompletion.create(
                    model=engine,
                    temperature=0,
                    max_tokens=max_tokens,
                    messages=self.dialog_history,
                )
                response_text = response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    model=engine,
                    # prompt=[question],
                    prompt=self.dialog_history_to_str(),
                    max_tokens=max_tokens,
                    temperature=0,
                    stop=stop_sign,
                )
                response_text = response['choices'][0]['text']
            self.num_tokens.append(response['usage']["total_tokens"])
            response_text = response_text.strip()

        output = f"S: {self.dialog_history[0]['content']}\n\nQ: {question}\n\nA: {response_text}\n"
        if verbose:
            print(output)
            self.print_cost()

        self.dialog_history.append({"role": "assistant", "content": response_text}, )

        if enable_pdb:
            import pdb;
            pdb.set_trace()

        if not (question in self.cache):
            datum = [{
                'pred': response_text,
                'query': question,
            }]
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(datum, self.output_file, mode='a')

        if only_response:
            return response_text
        return output


def main():
    raw_text = 'Hello, world. Here are two people with M.A. degrees from UT Austin. This is Mr. Mike.'
    nlp = NLP()
    sentences = nlp.sent_tokenize(raw_text)
    words = nlp.word_tokenize(sentences[0], lower=True)
    show_var(['sentences', 'words'])

    chat = Chatbot()
    query = 'What is the best way to learn Machine Learning?'
    response = chat.query_with_patience(query)


if __name__ == '__main__':
    main()
