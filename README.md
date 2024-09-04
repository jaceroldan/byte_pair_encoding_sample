# Byte Pair Encoding Algorithm

## Goal

Create a text tokenizer using a bottom-up approach as discussed in class. Use the dataset found in the Kaggle notebook in the lecture. Limit yourselves to a random sample of 1000 documents. For the sentence segmentation, use any algorithm, cite your references.

main:
https://www.kaggle.com/competitions/coleridgeinitiative-show-us-the-data/data
alternate (you didn't get it from here):
https://www.dropbox.com/scl/fi/ow2o40dpm8tu2ahg3muet/coleridgeinitiative-show-us-the-data.zip?rlkey=c5ffx9iol8cdkyymumtalbt9w&dl=0

Compare your tokenizer with the Wordpiece tokenizer found in the Hugging Face transformer library. Identify the number of tokens in each, where did the difference come from?

## Setup

1. Use Python3.10.
2. Create a virtual environment. `virtualenv <name-of-env>`
3. Activate virtual environment. `source ./<name-of-env/bin/activate`
4. Install dependencies. `pip install -r requirements.txt`
5. Download spacy language models for sentence segmentation. `python -m spacy download en_core_web_sm`

## Run
Simply run the statistics tester on test.py.

```bash
source ./<name-of-env>/bin/activate
python test.py
```

You may also adjust the values of k in the constructor for the MultiThreadedBytePairEncoder class.

```python
mt_bpe = MultiThreadedBytePairEncoder(samples=self.samples, k=500)
```
