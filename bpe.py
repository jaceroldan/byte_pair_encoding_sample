import random
import json
import pandas as pd
import os

# from sklearn.model_selection import train_test_split

df = None

if __name__ == '__main__':
    df = pd.read_csv('./datasets/train.csv')
    
    items = df.sample(n=1000)
    print(items.shape)
    print(items.head(n=10))
    n = 0
    while n < 1000:
        curr_path = os.path.join(os.getcwd(), 'datasets', 'train', items.iloc[n]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            print(curr_json)
        n += 1
