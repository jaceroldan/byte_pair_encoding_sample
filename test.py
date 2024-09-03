import os
import json
import pprint

import pandas as pd

from bpe import MultiThreadedBertTokenizer, MultiThreadedBytePairEncoder


class TestRunner:
    def __init__(self):
        df = None
        df = pd.read_csv('./datasets/train.csv')
        
        items = df.sample(n=1000, random_state=42)

        n = 0
        self.samples = []
        while n < 1000:
            curr_path = os.path.join(
                os.getcwd(), 'datasets', 'train', items.iloc[n]['Id'] + '.json')

            with open(curr_path, 'r') as file:
                curr_json = json.load(file)
                self.samples.append(''.join([cj['text'] for cj in curr_json]))
            n += 1
    
    def run_results(self):
        mt_bpe = MultiThreadedBytePairEncoder(samples=self.samples, k=200)
        bpe_tokens = mt_bpe.run()

        mt_wordpiece = MultiThreadedBertTokenizer(samples=self.samples)
        wordpiece_tokens = mt_wordpiece.run()

        only_in_bpe = bpe_tokens - wordpiece_tokens

        # Tokens unique to WordPiece
        only_in_wordpiece = wordpiece_tokens - bpe_tokens

        # Intersection of tokens (common tokens)
        intersection = bpe_tokens & wordpiece_tokens

        # Union of all tokens
        union = bpe_tokens | wordpiece_tokens

        # Percentage of intersection over union
        intersection_percentage = len(intersection) / len(union) * 100

        # Generate report
        report = {
            "Total BPE Tokens": len(bpe_tokens),
            "Total WordPiece Tokens": len(wordpiece_tokens),
            "Tokens only in BPE": list(only_in_bpe),
            "Tokens only in WordPiece": list(only_in_wordpiece),
            "Intersection Percentage": f"{intersection_percentage:.2f}%"
        }

        # Display the report
        pprint.pprint(report)


if __name__ == '__main__':
    runner = TestRunner()
    runner.run_results()
