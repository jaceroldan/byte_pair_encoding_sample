import pprint
import os
import re
import json
import pandas as pd

from collections import defaultdict

# Make selection of how to segment the sentences

df = None
corpus = '''Tokenization is the process of breaking down 
a sequence of text into smaller units called tokens,
which can be words, phrases, or even individual characters.
Tokenization is often the first step in natural languages processing tasks 
such as text classification, named entity recognition, and sentiment analysis.
The resulting tokens are typically used as input to further processing steps,
such as vectorization, where the tokens are converted
into numerical representations for machine learning models to use.'''



def merge(vocab, best_pair):
    new_vocab = defaultdict(int)

    new_token = re.escape(' '.join(best_pair))
    p = re.compile(r'(?<!\S)' + new_token + r'(?!\S)')

    for word in vocab:
        altered_word = p.sub(''.join(best_pair), word)
        new_vocab[altered_word] = vocab[word]

    return new_vocab


def count_pairs(vocab):
    pairs = defaultdict(int)

    for word in vocab:
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += 1

    return pairs


def extract_vocab(s):
    vocab = defaultdict(int)

    for word in s.split():
        vocab[' '.join(word) + ' </w>'] += 1

    return vocab


def bpe(s, k=200):
    # TODO: Figure out which algorithm to use for sentence extraction.
    vocab = extract_vocab(s)

    for i in range(k):
        pairs = count_pairs(vocab)
        if len(pairs) == 0:
            print('max iters at k = {}'.format(i))
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge(vocab, best_pair)

    return vocab


if __name__ == '__main__':
    df = pd.read_csv('./datasets/train.csv')
    
    items = df.sample(n=1000)

    n = 0
    samples = []
    while n < 1000:
        curr_path = os.path.join(os.getcwd(), 'datasets', 'train', items.iloc[n]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            # TODO: Include the cj['section_title'] in the parsing?
            samples.append(''.join([cj['text'] for cj in curr_json]))
        n += 1

    pprint.pprint(bpe(corpus, k=240))

