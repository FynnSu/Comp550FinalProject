import sys
import json
import argparse
import os.path as osp
import os
import requests
import requests.auth
import time
import pandas as pd

REDDIT_TOKEN = ''
REDDIT_SECRET = ''
REDDIT_USERNAME = ''
REDDIT_PASSWORD = ''



def get_current_time_utc():
    return int(round(time.time()))

def get_token():
    client_auth = requests.auth.HTTPBasicAuth(REDDIT_TOKEN, REDDIT_SECRET)
    post_data = {"grant_type": "password", "username": REDDIT_USERNAME, "password": REDDIT_PASSWORD}
    headers = {"User-Agent": f"CollectRedditData/0.1 by {REDDIT_USERNAME}"}
    response = requests.post("https://www.reddit.com/api/v1/access_token", auth=client_auth, data=post_data, headers=headers)
    return response.json()['access_token']

def main(args):
    if not osp.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    t = get_current_time_utc()

    token = get_token()
    headers = {"User-Agent": f"CollectRedditData/0.1 by {REDDIT_USERNAME}", "Authorization": "bearer {}".format(token)}

    # Go through each subreddit subfolder of input_dir and for all json files in it extract the json data
    for subreddit in sorted(os.listdir(args.input_dir)):

        if get_current_time_utc() - t > 3000:
            t = get_current_time_utc()
            token = get_token()
            headers = {"User-Agent": f"CollectRedditData/0.1 by {REDDIT_USERNAME}", "Authorization": "bearer {}".format(token)}

        print("Processing {}".format(subreddit))
        sys.stdout.flush()
        subreddit_dir = osp.join(args.input_dir, subreddit)
        if not osp.isdir(subreddit_dir):
            continue

        # Create output csv file
        output_file = osp.join(args.output_dir, "{}.csv".format(subreddit))
        if osp.isfile(output_file):
            print("Skipping {}".format(subreddit))
            sys.stdout.flush()
            continue
        unique_ids = set()
        for filename in sorted(os.listdir(subreddit_dir)):
            if not filename.endswith(".json"):
                continue
            # print("Processing {}".format(filename))
            with open(osp.join(subreddit_dir, filename), 'r') as f:
                data = json.load(f)

            post_ids = []
            for post in data:
                id = post['id']
                if not id.startswith("t3_"):
                    id = "t3_{}".format(id)
                post_ids.append(id)
            
            # Get the posts
            response = requests.get(f"https://oauth.reddit.com/r/{subreddit}/api/info.json?id={','.join(post_ids)}", headers=headers)
            if response.status_code != 200:
                print("Error getting posts for {}".format(subreddit))
                sys.stdout.flush()
                continue
            
            posts = [y['data'] for y in response.json()['data']['children'] if y['data']['id'] not in unique_ids]
            unique_ids.update([y['id'] for y in posts])
            
            df = pd.DataFrame(posts, columns=['id', 'title', 'selftext', 'url', 'created_utc', 'score', 'upvote_ratio', 'num_comments', 'subreddit', 'author'])
            df = df.set_index('id')

            # Save to csv
            df.to_csv(output_file, mode='a', header=not osp.exists(output_file))

        print("Done processing {}, {} posts found".format(subreddit, len(unique_ids)))
        sys.stdout.flush()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get recent posts from a subreddit')
    parser.add_argument('input_dir', type=str, help='Folder to start from')
    parser.add_argument('output_dir', type=str, help='Folder to end at')
    args = parser.parse_args()

    main(args)