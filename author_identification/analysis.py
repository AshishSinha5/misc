import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

label_code = {
    'EAP': 0,
    'HPL': 1,
    'MWS': 2
}

author_code = {v: k for k, v in label_code.items()}
stop_words = set(stopwords.words("english"))

def plot_hist(au, feat, bins=100):
    plt.hist(df[df['author'] == au][feat], bins=bins)
    plt.xlabel(author_code[au])
    plt.ylabel(feat)
    plt.savefig('plots/{}_{}.png'.format(au, feat))
    plt.show()

def get_puncts(x):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    return count(x, string.punctuation)

def get_words(x):
    return len([w for w in x.split(' ')])


def get_ave_len(x):
    return np.mean([len(w) for w in word_tokenize(x)])


def num_stop_words(x):
    return len([w for w in word_tokenize(x) if w in stop_words])


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df['author'] = pd.Series(list(map(lambda x: label_code[x], df['author'])))
    df['len'] = df['text'].apply(len)
    df['puncts'] = df['text'].apply(get_puncts)
    df['num_words'] = df['text'].apply(get_words)
    df['ave_len_word'] = df['text'].apply(get_ave_len)
    df['num_stop_words'] = df['text'].apply(num_stop_words)
    plot_hist(0, 'len')
    plot_hist(1, 'len')
    plot_hist(2, 'len')
    plot_hist(0, 'puncts', 15)
    plot_hist(1, 'puncts', 15)
    plot_hist(2, 'puncts', 15)
    plot_hist(0, 'num_words')
    plot_hist(1, 'num_words')
    plot_hist(2, 'num_words')
    plot_hist(0, 'ave_len_word')
    plot_hist(1, 'ave_len_word')
    plot_hist(2, 'ave_len_word')
    plot_hist(0, 'num_stop_words', 30)
    plot_hist(1, 'num_stop_words', 30)
    plot_hist(2, 'num_stop_words', 30)
