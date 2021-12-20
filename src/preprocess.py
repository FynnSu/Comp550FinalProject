import argparse
import os.path as osp
import json
import sys
from os import walk
import pandas as pd
from tqdm import tqdm
import os

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    return parser.parse_args()


def main():

    args = parse()
    os.makedirs(args.o, exist_ok=True)
    dirnames = next(walk(args.i), (None, None, []))[1]
    for subreddit in tqdm(dirnames):
        filenames = next(walk(osp.join(args.i, subreddit)), (None, None, []))[2]

        df = pd.DataFrame(columns=["ratio", "title", "text", "score", "url"])
        for file in filenames:

            with open(osp.join(args.i, subreddit, file), "r") as fi:
                sample = json.load(fi)
                ratios, titles, texts, scores, urls = [], [], [], [], []

                for post in sample:
                    ratios.append(post.get("upvote_ratio", "N/A"))
                    titles.append(post.get("title", "N/A"))
                    texts.append(post.get("selftext", "N/A"))
                    scores.append(post.get("score", "N/A"))
                    urls.append(post.get("url", "N/A"))
                    # df = df.append([post["upvote_ratio"], post.get("title", ""), post.get("selftext", "")])
                sampledf = pd.DataFrame({"ratio":ratios,"title":titles,"text":texts, "score":scores, "url": urls})

                df = pd.concat([df,sampledf], ignore_index=True)

        df.drop_duplicates(inplace=True)
        # print(df)
        # with open(osp.join(args.o,subreddit+".json"),"w") as fo:
        #     df.to_csv(fo, sep="\t")
        df.to_csv(osp.join(args.o,subreddit+".tsv"), sep="\t")


if __name__ == '__main__':
    main()
