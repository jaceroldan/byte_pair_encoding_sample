import json
import pandas as pd
import os

# from sklearn.model_selection import train_test_split

df = None

def extract_unique_chars(s):
    count_dict = {}
    for c in s:
        if type(count_dict.get(c, None)) == int:
            count_dict[c] += 1
        else:
            count_dict[c] = 0

    return count_dict


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

    print(extract_unique_chars(samples[0]))
    
