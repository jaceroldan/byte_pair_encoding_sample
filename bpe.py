import time
import os
import re
import json
import pandas as pd

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import spacy
from transformers import BertTokenizer


class AbstractBytePairEncoder:
    def __init__(self, samples, k=200, beta=0.5, segment_sentence=True, *args, **kwargs):
        self.samples = samples
        self.k = k
        self.compression_times = {}
        self.compression_ratios = {}
        self.beta = beta
        self.heaps_law_k = {}
        self.num_types = {}
        self.num_tokens = {}
        self.segment_sentence = segment_sentence
        if segment_sentence:
            self.nlp = spacy.load('en_core_web_sm')

    def segment_sentence_using_spacy(self, text):
        # Process the text using spaCy
        doc = self.nlp(text)
        
        # Extract sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences

    def compute_compression_ratio(self, original, compressed):
        original_size = sum(len(word.split()) for word in original.split())
        compressed_size = sum(len(word.split()) for word in compressed)
        return original_size / compressed_size if compressed_size != 0 else float('inf')

    def compute_heaps_law(self, vocab):
        num_tokens = sum(vocab.values())
        num_types = len(vocab)
        k = num_types / (num_tokens ** self.beta)
        return k, num_tokens, num_types
    
    def merge(self, vocab, best_pair):
        new_vocab = defaultdict(int)

        # Escape special characters in the best pair and join them with a space
        new_token = re.escape(' '.join(best_pair))

        # Check if the pair contains special characters that could cause issues
        if "\\" in best_pair[0] or "\\" in best_pair[1]:
            print(f"Skipping problematic pair: {best_pair}")
            return vocab  # Skip this problematic pair

        # Compile the regular expression, escaping any special characters in the new token
        try:
            p = re.compile(r'(?<!\S)' + new_token + r'(?!\S)')
        except re.error as e:
            print(f"Regex compilation error for pair {best_pair}: {e}")
            raise e

        for word in vocab:
            try:
                # Perform the substitution, making sure the escape sequences are handled correctly
                altered_word = p.sub(''.join(best_pair), word)
                new_vocab[altered_word] = vocab[word]
            except re.error as e:
                print(f"Regex substitution error for word {word} and pair {best_pair}: {e}")
                raise e

        return new_vocab

    def count_pairs(self, vocab):
        pairs = defaultdict(int)

        for word in vocab:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += 1

        return pairs

    def extract_vocab(self, s):
        vocab = defaultdict(int)

        for word in s.split():
            vocab[' '.join(word) + ' </w>'] += 1

        return vocab

    def bpe(self, s):
        vocab = self.extract_vocab(s)
        original_vocab = vocab.copy()

        for i in range(self.k):
            pairs = self.count_pairs(vocab)
            print(f"Iteration {i}, Vocab size: {len(vocab)}")
            if len(pairs) == 0:
                print('No more pairs to merge at iteration:', i)
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge(vocab, best_pair)

        compression_ratio = self.compute_compression_ratio(
            ' '.join(original_vocab.keys()), ' '.join(vocab.keys()))
        print(f"Compression Ratio at iteration {i}: {compression_ratio}")
        return vocab, compression_ratio

    def process_sample(self, sample):
        """
        Process a single sample which could be a list of sentences (if segmentation is enabled).
        """
        # Segment the sample into sentences if segmentation is enabled
        if self.segment_sentence:
            sentences = self.segment_sentence_using_spacy(sample)
        else:
            sentences = [sample]  # Treat the whole sample as one sentence if not segmenting

        # Process each sentence with BPE and combine vocabularies
        combined_vocab = defaultdict(int)
        for sentence in sentences:
            sentence_vocab, _ = self.bpe(sentence)
            for word, count in sentence_vocab.items():
                combined_vocab[word] += count

        return combined_vocab

class SingleThreadBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        print('Running single-threaded BPE tokenizer...')
        token_set = []
        start = time.time()
        for i, sample in enumerate(self.samples):
            item_start = time.time()
            print('Processing item:', i + 1)
            result = self.process_sample(sample)  # Process the sample with segmentation
            token_set.append(result)
            self.compression_ratios[i] = self.compute_compression_ratio(sample, ' '.join(result.keys()))
            item_end = time.time()
            self.compression_times[i] = f'{item_end - item_start:.8f}'
            k, num_tokens, num_types = self.compute_heaps_law(result)
            self.heaps_law_k[i] = k
            self.num_tokens[i] = num_tokens
            self.num_types[i] = num_types
        end = time.time()
        print(f"Processing time: {end - start} seconds")
        return token_set


class MultiThreadedBytePairEncoder(AbstractBytePairEncoder):
    def run(self):
        print('Running multi-threaded BPE tokenizer...')
        token_set = []
        start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.process_sample, sample) for sample in self.samples]
            for i, future in enumerate(as_completed(futures)):
                item_start = time.time()
                print('Processing item:', i + 1)
                result = future.result()
                token_set.append(result.keys())
                self.compression_ratios[i] = self.compute_compression_ratio(self.samples[i], ' '.join(result.keys()))
                item_end = time.time()
                k, num_tokens, num_types = self.compute_heaps_law(result)
                self.heaps_law_k[i] = k
                self.num_tokens[i] = num_tokens
                self.num_types[i] = num_types
                self.compression_times[i] = f'{item_end - item_start:.8f}'

        end = time.time()
        print(f"Processing time: {end - start} seconds")
        return token_set


class AbstractWordPieceTokenizer:
    def __init__(self, samples, segment_sentence=True, *args, **kwargs):
        self.samples = samples
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.compression_times = {}
        self.compression_ratios = {}
        self.apply_sentence_segmentation = segment_sentence
        self.nlp = spacy.load('en_core_web_sm')
    
    def tokenize(self, sample):
        original_size = len(sample.split())

        tokens = self.tokenizer.tokenize(sample)
        compressed_size = len(tokens)

        compression_ratio = (
            original_size / compressed_size
            if compressed_size != 0
            else float('inf')
        )

        return tokens, compression_ratio

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
            item_start = time.time()
            print('item:', i + 1)
            results = self.tokenize(sample)
            tokens = results[0]
            self.compression_ratios[i] = results[1]
            token_set.append(tokens)
            item_end = time.time()
            self.compression_times[i] = f'{item_end - item_start:.8f}'

            
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
                item_start = time.time()
                print('item:', i + 1)
                results = future.result()
                result = results[0]
                token_set.append(result)
                self.compression_ratios[i] = results[1]
                item_end = time.time()
                self.compression_times[i] = f'{item_end - item_start:.8f}'
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
    
    n = 10
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
