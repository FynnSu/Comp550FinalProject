
import sys
import json
import argparse
import os.path as osp
import os
import requests
import requests.auth
import time


def get_current_time_utc():
    return int(round(time.time()))


def get_100_posts(subreddit, before_time):
    url = f'https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&size=100&sort=desc&sort_type=created_utc&score=>50&before={before_time}'

    while True:
        response = requests.get(url)
        if response.status_code == 200:
            break
        print('Rate limit exceeded, sleeping for 5 seconds')
        sys.stdout.flush()
        time.sleep(5)
    
    try:
        data = response.json()['data']
    except:
        print(f'Error getting data, response code: {response.status_code}, response: {response.json()}')
        sys.stdout.flush()
        return []
    return data

def main2(args):
    with open(args.subreddit_list_file, 'r') as f:
        subreddits = f.read().splitlines()

    if not osp.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.isdir(args.log_dir):
        os.mkdir(args.log_dir)

    if args.cur_sub_time_file != '':
        with open(args.cur_sub_time_file, 'r') as f:
            cur_sub_time_dict = json.load(f)
        
    else:
        cur_sub_time_dict = {x: get_current_time_utc() for x in subreddits}

    for subreddit in subreddits:
        if not osp.isdir(osp.join(args.output_dir, subreddit)):
            os.mkdir(osp.join(args.output_dir, subreddit))

    file_num_for_sub = {x: 0 for x in subreddits}

    for subreddit in subreddits:
        while osp.isfile(osp.join(args.output_dir, subreddit, '{}.json'.format(file_num_for_sub[subreddit]))):
            file_num_for_sub[subreddit] += 1

    post_count = 0

    while True:
        print('Downloading posts. Current count: {}'.format(post_count))

        to_remove = []
        for subreddit in subreddits:
            print('Getting posts from {}'.format(subreddit))

            posts = get_100_posts(subreddit, cur_sub_time_dict[subreddit])
            if len(posts) == 0:
                print('No more posts found for {}. Removing subreddit from list.'.format(subreddit))
                to_remove.append(subreddit)
                continue
            cur_sub_time_dict[subreddit] = int(posts[-1]['created_utc'])
            
            with open(osp.join(args.output_dir, subreddit, f'{file_num_for_sub[subreddit]}.json'), 'w') as f:
                json.dump(posts, f)
            
            file_num_for_sub[subreddit] += 1
            
            sys.stdout.flush()
        
        subreddits = [x for x in subreddits if x not in to_remove]
        post_count += 100

        with open(osp.join(args.log_dir, 'cur_sub_time.json'), 'w') as f:
            json.dump(cur_sub_time_dict, f)

        if len(subreddits) == 0:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get recent posts from a subreddit')
    parser.add_argument('subreddit_list_file', type=str, help='File containing list of subreddits')
    parser.add_argument('output_dir', type=str, help='File to write output to')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to write logs to')
    parser.add_argument('--limit', type=int, default=1000, help='Number of posts to get')
    parser.add_argument('--cur_sub_time_file', type=str, default="", help='File containing current time for each subreddit')

    args = parser.parse_args()

    main2(args)