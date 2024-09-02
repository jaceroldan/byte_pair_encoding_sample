import time
import pprint
import os
import re
import json
import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


df = None
# test_corpus = '''Tokenization is the process of breaking down 
# a sequence of text into smaller units called tokens,
# which can be words, phrases, or even individual characters.
# Tokenization is often the first step in natural languages processing tasks 
# such as text classification, named entity recognition, and sentiment analysis.
# The resulting tokens are typically used as input to further processing steps,
# such as vectorization, where the tokens are converted
# into numerical representations for machine learning models to use.'''


class AbstractBytePairEncoder:
    def __init__(self, samples, k=200, *args, **kwargs):
        self.samples = samples
        self.k = k

    def merge(self, vocab, best_pair):
        new_vocab = defaultdict(int)

        new_token = re.escape(' '.join(best_pair))
        p = re.compile(r'(?<!\S)' + new_token + r'(?!\S)')

        for word in vocab:
            altered_word = p.sub(''.join(best_pair), word)
            new_vocab[altered_word] = vocab[word]

        return new_vocab


    def count_pairs(self, vocab):
        pairs = defaultdict(int)

        for word in vocab:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += 1

        return pairs


    def extract_vocab(self, s):
        vocab = defaultdict(int)

        for word in s.split():
            vocab[' '.join(word) + ' </w>'] += 1

        return vocab

    def bpe(self, s):
        # TODO: Figure out which algorithm to use for sentence extraction.
        vocab = self.extract_vocab(s)

        for i in range(self.k):
            pairs = self.count_pairs(vocab)
            if len(pairs) == 0:
                print('max iters at k = {}'.format(i))
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge(vocab, best_pair)

        return vocab
    
    def run(self, k=200):
        return NotImplementedError(
            'Please use the single-threaded or parallelized subclasses '
            'to run this function!'
        )


class SingleThreadBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        start = time.time()
        for i, sample in enumerate(self.samples):
            # without pre-tokenization 
            print('item:', i)
            pprint.pprint(self.bpe(sample))
        end = time.time()
        print(end - start, 'seconds')


class MultiThreadedBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.bpe, sample) for sample in self.samples]
            for i, future in enumerate(as_completed(futures)):
                # without pre-tokenization 
                print('item:', i)
                pprint.pprint(future.result())

        end = time.time()
        print(end - start, 'seconds')


if __name__ == '__main__':
    df = pd.read_csv('./datasets/train.csv')
    
    items = df.sample(n=1000, random_state=42)

    n = 0
    samples = []
    while n < 1000:
        curr_path = os.path.join(
            os.getcwd(), 'datasets', 'train', items.iloc[n]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            # TODO: Include the cj['section_title'] in the parsing?
            samples.append(''.join([cj['text'] for cj in curr_json]))
        n += 1  

    # st_bpe = SingleThreadBytePairEncoder(samples=samples, k=200)
    # st_bpe.run()
    
    mt_bpe = MultiThreadedBytePairEncoder(samples=samples, k=200)
    mt_bpe.run()
