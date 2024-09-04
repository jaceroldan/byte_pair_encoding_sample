import os
import json
import pprint

import pandas as pd

from bpe import MultiThreadedBertTokenizer, MultiThreadedBytePairEncoder


class TestRunner:
    def __init__(self):
        df = None
        df = pd.read_csv('./datasets/train.csv')
        
        n = 10
        items = df.sample(n=n, random_state=42)

        i = 0
        self.samples = []
        while i < n:
            curr_path = os.path.join(
                os.getcwd(), 'datasets', 'train', items.iloc[i]['Id'] + '.json')

            with open(curr_path, 'r') as file:
                curr_json = json.load(file)
                self.samples.append(''.join([cj['text'] for cj in curr_json]))
            i += 1
    
    def run_results(self):
        mt_bpe = MultiThreadedBytePairEncoder(samples=self.samples, k=200)
        bpe_tokens_list = mt_bpe.run()

        mt_wordpiece = MultiThreadedBertTokenizer(samples=self.samples)
        wordpiece_tokens_list = mt_wordpiece.run()

        overall_report = []

        for i in range(len(self.samples)):
            bpe_tokens = set([token.replace('</w>', '') for token in bpe_tokens_list[i]])
            wordpiece_tokens = set(wordpiece_tokens_list[i])

            # Tokens unique to BPE
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
            overall_report.append(report)

            # Display the report
            pprint.pprint(report)


if __name__ == '__main__':
    runner = TestRunner()
    runner.run_results()
