import time
import pprint
import os
import re
import json
import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


from transformers import BertTokenizer


# TODO: Extract stats for compression ratio, Heaps' law
class AbstractBytePairEncoder:
    def __init__(self, samples, k=200, beta=0.5, *args, **kwargs):
        self.samples = samples
        self.k = k
        self.compression_time = None
        self.compression_ratio = None
        self.beta = beta
        self.heaps_law_k = None
        self.num_types = None
        self.num_tokens = None

    def compute_compression_ratio(self, original, compressed):
        """
        For the computation of data compression ratio.
        We compute the ratio of the size of the original vocab to the size
        of the vocab of the result.

        Reference: 
        """
        original_size = sum(len(word.split()) for word in original.split())
        compressed_size = sum(len(word.split()) for word in compressed)
        return original_size / compressed_size

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
        # TODO: Figure out which algorithm to use for sentence segmentation.
        vocab = self.extract_vocab(s)

        for i in range(self.k):
            pairs = self.count_pairs(vocab)
            if len(pairs) == 0:
                print('max iters at k = {}'.format(i))
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge(vocab, best_pair)

        return vocab
    
    def run(self):
        return NotImplementedError(
            'Please use the single-threaded or parallelized subclasses '
            'to run this function!'
        )


class SingleThreadBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        print('Running single-threaded BPE tokenizer...')
        token_set = []
        start = time.time()
        for i, sample in enumerate(self.samples):
            # without pre-tokenization 
            print('item:', i + 1)
            result = self.bpe(sample)
            token_set.append(result)
        end = time.time()
        print(end - start, 'seconds')
        return token_set


class MultiThreadedBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        print('Running multi-threaded BPE tokenizer...')
        token_set = []
        start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.bpe, sample) for sample in self.samples]
            for i, future in enumerate(as_completed(futures)):
                # without pre-tokenization 
                print('item:', i + 1)
                result = future.result().keys()
                token_set.append(result)

        end = time.time()
        print(end - start, 'seconds')
        return token_set


class AbstractWordPieceTokenizer:
    def __init__(self, samples, *args, **kwargs):
        self.samples = samples
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def tokenize(self, sample):
        tokens = self.tokenizer.tokenize(sample)
        return tokens

    def run(self):
        return NotImplementedError(
            'Please use the single-threaded or parallelized subclasses '
            'to run this function!'
        )


class SingleThreadedBertTokenizer(AbstractWordPieceTokenizer):
    def run(self):
        print('Running single threaded WordPiece tokenizer...')
        token_set = []
        start = time.time()
        for i, sample in enumerate(self.samples):
            print('item:', i + 1)
            tokens = self.tokenize(sample)
            token_set.append(tokens)
            
        end = time.time()
        print(end - start, 'seconds')
        return token_set


class MultiThreadedBertTokenizer(AbstractWordPieceTokenizer):
    def run(self):
        print('Running multi-threaded WordPiece tokenizer...')
        start = time.time()
        token_set = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.tokenize, sample) for sample in self.samples]
            for i, future in enumerate(as_completed(futures)):
                print('item:', i + 1)
                result = future.result()
                token_set.append(result)
        end = time.time()
        print(end - start, 'seconds')
        return token_set



if __name__ == '__main__':
    test_corpus = '''Tokenization is the process of breaking down 
        a sequence of text into smaller units called tokens,
        which can be words, phrases, or even individual characters.
        Tokenization is often the first step in natural languages processing tasks 
        such as text classification, named entity recognition, and sentiment analysis.
        The resulting tokens are typically used as input to further processing steps,
        such as vectorization, where the tokens are converted
        into numerical representations for machine learning models to use.'''

    df = None
    df = pd.read_csv('./datasets/train.csv')
    
    n = 1000
    items = df.sample(n=n, random_state=42)

    i = 0
    samples = []
    while i < n:
        curr_path = os.path.join(
            os.getcwd(), 'datasets', 'train', items.iloc[i]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            # TODO: Include the cj['section_title'] in the parsing?
            samples.append(''.join([cj['text'] for cj in curr_json]))
        i += 1  

    mt_bpe = MultiThreadedBytePairEncoder(samples=samples, k=200)
    mt_bpe.run()

    mt_wordpiece = MultiThreadedBertTokenizer(samples=samples)
    mt_wordpiece.run()
