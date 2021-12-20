import nltk
import argparse
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import os.path as osp
from tqdm import tqdm
from gensim.models import Word2Vec
from os import walk
from sklearn.manifold import TSNE
# import plotly.express as px
# import umap
# from sklearn.feature_extraction.text import CountVectorizer


# Get list of English stopwords
stop_words = set(stopwords.words('english') + ["removed", "deleted", "nan", "http"])

def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', type=str, required=True)
    parser.add_argument('-i', type=str, default="final_data_with_ratios")
    return parser.parse_args()


def tokenize(post, stop_words, stemmer):
    words = [stemmer.stem(word).strip() for word in nltk.word_tokenize(post) if word not in stop_words and
             stemmer.stem(word) not in stop_words and word.isalnum() and not stemmer.stem(word).isdecimal()]
    return words


def remove_underscores(posts):
    new_posts = []
    for p in posts:
        new_post = []
        for word in p:
            new_post += word.split('_')

        new_posts.append(new_post)
    return new_posts


def get_frequent_words(posts, threshold=5):
    word_counts = {}
    for post in posts:
        for word in post:
            word_counts[word] = word_counts.get(word, 0) + 1

    frequent_words = []
    for k, v in word_counts.items():
        if v >= threshold:
            frequent_words.append(k)

    return frequent_words


def limit_to_frequent_words(posts):
    freq_words = set(get_frequent_words(posts))
    new_posts = []
    for p in posts:
        new_post = []
        for word in p:
            if word in freq_words:
                new_post.append(word)

        new_posts.append(new_post)
    return new_posts


def main():
    args = parse()

    filenames = next(walk(osp.join(args.i)), (None, None, []))[2]
    # filenames = ['AskMen.csv', 'politics.csv']
    pbar = tqdm(filenames)
    for file in pbar:
        pbar.set_description("Processing %s" % file)

        posts = pd.read_csv(osp.join(args.i, file), lineterminator='\n')
        weight_df = pd.read_csv(osp.join("weights", file), index_col=0)
        # posts = pd.read_csv(osp.join("final_data", args.r + ".csv"), lineterminator='\n')
        # weight_df = pd.read_csv(osp.join("weights", args.r + ".csv"), index_col=0)
        # Set up a stemmer

        print('len weight_df: ', len(weight_df))

        stemmer_ps = PorterStemmer()
        # Get list of English stopwords
        # stop_words = set(stopwords.words('english'))

        token_data = []
        for post in list(posts["title"].astype(str)+" "+posts["selftext"].astype(str)):
            token_data.append(tokenize(post, stop_words, stemmer_ps))

        token_data = remove_underscores(token_data)
        token_data = limit_to_frequent_words(token_data)

        # print(len(set(weight_df['feature'])), "Unique tokens in weight csv")

        # print('first 20 tokens', token_data[:20])

        # print(list(weight_df["feature"]))
        model = Word2Vec(token_data, min_count=0)
        # print(model)
        # print(model.wv.rank_by_centrality(["men","women","sex","hair","threesom"]))
        vector_list = model.wv.vectors

        # Create a list of the words corresponding to these vectors
        words_filtered = model.wv.index_to_key

        print('n_words_filtered', len(words_filtered))

        # print(words_filtered)

        # print('Len filtered', len(words_filtered))

        # index = weight_df.index
        # print(words_filtered)
        # print(index[weight_df["feature"]== "i"][0])
        # not_found_count = 0
        # for word in words_filtered:
        #     condition = weight_df["feature"] == word
        #     indices = index[condition]
        #     if len(indices) == 0:
        #         print(word)
        #         not_found_count += 1
        #
        # print(not_found_count)

        # Zip the words together with their vector representations
        word_vec_zip = zip(words_filtered, vector_list)

        # Cast to a dict so we can turn it into a DataFrame
        word_vec_dict = dict(word_vec_zip)
        df = pd.DataFrame.from_dict(word_vec_dict, orient='index')
        pd.DataFrame(model.wv.rank_by_centrality(df.index, use_norm=False)).to_csv(f"centralities/{file}")
        # Initialize t-SNE
        tsne = TSNE(n_components=3, init='random', random_state=42, perplexity=20)
        index = weight_df.index
        # weights = [index[weight_df["feature"] == word][0] for word in words_filtered if weight_df['feature'].str.contains(word).any()]

        weight_dict = {v: k for k, v in weight_df.to_dict()['feature'].items()}
        tsne_df = pd.DataFrame(np.hstack([np.array(tsne.fit_transform(df)), np.expand_dims(np.array(words_filtered), axis=-1)]), columns=["x", "y", "z", "words"])

        tsne_df['weight'] = [weight_dict.get(word, 0) for word in tsne_df['words']]
        tsne_df = tsne_df[tsne_df['weight'] != 0]
        # tsne_df['weight'] = weights
        tsne_df.to_csv(f"embedding/{file}")
        # print(tsne_df)
        # fig = px.scatter_3d(x=tsne_df["x"].astype(float), y=tsne_df["y"].astype(float), z=tsne_df["z"].astype(float), hover_name=tsne_df["words"], color=weights)
        # fig.show()


if __name__ == '__main__':
    main()
