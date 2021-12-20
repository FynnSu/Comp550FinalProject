import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import argparse
import os.path as osp
from os import walk
import json
import pandas as pd
from tqdm import tqdm

# Get list of English stopwords
stop_words = set(stopwords.words('english') + ["removed", "deleted", "nan", "http"])
# Install necessary packages for NLTK if they don't already exist
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="final_data")
    return parser.parse_args()


def preprocess_forge(alphanum=False, stop_words=None, stemmer=None):
    """
    Creates a preprocessing pipeline that always tokenizes words and is otherwise configured based on forge parameters
    :param alphanum: (bool) keep only alpha numerical entries if True
    :param stop_words: (list) a list of stopwords to remove from the corpus
    :param stemmer: (stemmer) a nltk stemmer that will be used to stem entries
    :return: (func) a preprocessing pipeline
    """

    def preprocess(review):
        """
        Preprocessing pipeline
        :param review: (str) the sample to be preprocessed
        :return: (str) preprocessed sample
        """
        # words = nltk.word_tokenize(review)
        # if alphanum:
        #     # keep only alpha numerical words
        #     words = [word for word in words if word.isalnum()]
        # if stop_words:
        #     # remove all stop_words from the sample
        #     print(f"before: {'i' in words}")
        #     words = [word for word in words if word not in stop_words]
        #     print("i" in words)
        # if stemmer:
        #     # stem the words in the sample
        #     words = [stemmer.stem(word) for word in words if word.isalnum()]
        #     print("i" in words)

        # words = [word for word in words if word not in stop_words]
        # words = [stemmer.stem(word) for word in words if word.isalnum()]
        words = [stemmer.stem(word) for word in nltk.word_tokenize(review) if word not in stop_words and stemmer.stem(word) not in stop_words and word.isalnum() and not stemmer.stem(word).isdecimal()]

        return " ".join(words)

    # return the preprocessing function
    return preprocess

def tokenize(post, stop_words, stemmer):
    words = [stemmer.stem(word).strip() for word in nltk.word_tokenize(post) if word not in stop_words and stemmer.stem(word) not in stop_words and word.isalnum() and not stemmer.stem(word).isdecimal()]
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
    # set a seed for reproducibility
    seed = 42

    args = parse()

    filenames = next(walk(osp.join(args.i)), (None, None, []))[2]
    filenames = ['AskMen.csv', 'politics.csv']

    for file in tqdm(filenames):
        print(file)
        # print(file)
        posts = pd.read_csv(osp.join(args.i, file), lineterminator='\n')

        data = list(posts["title"].astype(str) + " " + posts["selftext"].astype(str))

        # Set up a stemmer
        stemmer_ps = PorterStemmer()

        token_data = []
        for post in list(data):
            token_data.append(tokenize(post, stop_words, stemmer_ps))

        token_data = remove_underscores(token_data)
        token_data = limit_to_frequent_words(token_data)
        token_data = [" ".join(post) for post in token_data]

        unique_words = set()
        for post in token_data:
            for token in post.split():
                unique_words.add(token)

        print(len(unique_words), "Unique tokens found")

        data_labels = posts["upvote_ratio"]  > np.median(posts["upvote_ratio"])

        # create the train test split with the label as the stratification criterion
        # x_train, x_test, y_train, y_test = train_test_split(token_data, data_labels, random_state=seed, shuffle=True,
        #                                                     stratify=data_labels)

        # create the pipeline that will be used to preprocess words and vectorize them (vect), then weigh their frequencies
        # (tfidf) and finally classify them (clf) using a logistic regression model
        # This approach is adapted from https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html
        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(verbose=0, random_state=seed, C=5, penalty='l2', max_iter=1000)),
        ])


        # Define parameter grid
        parameters = {
            # 'vect__max_df': np.logspace(-5, 0, 5, base=np.e),
            # 'vect__max_df': [0.5],
            'vect__ngram_range': (
                # (1, 1),
                (1, 1),
                # (1, 3),
            ),
            'vect__min_df': [0,],
            'vect__token_pattern': [r"(?u)\b\w+\b"]
            # 'vect__preprocessor': (
            #     # preprocess_forge(stop_words=stop_words),
            #     preprocess_forge(stop_words=stop_words, stemmer=stemmer_ps),),
        }

        # run 5-fold-cross-validated grid search on the parameter grid to find the best configurations of parameters
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=0)
        grid_search.fit(token_data, data_labels)

        # cv = CountVectorizer(ngram_range=(1, 1), min_df=0, token_pattern=r"(?u)\b\w+\b")
        # result = cv.fit_transform(token_data, data_labels)

        # print(list(cv.get_feature_names()))
        #
        # print(result.shape)

        # find most important features
        features = np.array(grid_search.best_estimator_["vect"].get_feature_names())
        print(len(features))
        weights = grid_search.best_estimator_["clf"].coef_[0]
        # sorted_index = np.argsort(grid_search.best_estimator_["clf"].coef_)[::-1][0]
        # top = sorted_index[:10]
        # bottom = sorted_index[-10:]
        # fig, ax = plt.subplots(2)
        # ax[0].bar(range(10), weights[bottom], tick_label=features[bottom], color="g")
        # ax[1].bar(range(10), weights[top], tick_label=features[top], color="r")
        # plt.show()
        # print(features[top])
        df = pd.DataFrame(features, index=weights, columns=["feature"])
        df = df.sort_index()
        df.to_csv(osp.join("weights", str(file)))

    # return
    #
    # # print the best parameters
    # best_parameters = grid_search.best_estimator_.get_params()
    # print("\nBest parameters:")
    # for param_name in sorted(parameters.keys()):
    #     # Find the name of the best preprocessing pipeline
    #     if param_name == "vect__preprocessor":
    #         prep_names = ["None", "alpha numerical only", "remove stop words", "stem", "remove stop words and stem"]
    #         print(
    #             f"\t{param_name:20s} = {prep_names[grid_search.param_grid[param_name].index(best_parameters[param_name])]}")
    #         continue
    #     print(f"\t{param_name:20s} = {best_parameters[param_name]}")
    #
    # # score the model
    # predicted_labels = grid_search.predict(x_test)
    # print(f"\n confusion matrix: \n{confusion_matrix(y_test, predicted_labels)}")
    # print(f"\nThe model achieved an accuracy of {accuracy_score(y_test, predicted_labels)}\n")
    # print(classification_report(y_test, predicted_labels, target_names=["uctv", "ctv"]))
    # # plot the results
    # # plot_result_grid(grid_search.cv_results_["mean_test_score"])


if __name__ == "__main__":
    main()
