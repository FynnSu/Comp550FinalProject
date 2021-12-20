import sys
import praw

CLIENT_SECRET = ''
CLIENT_ID = ''

def main():
    filename = sys.argv[1]
    with open(filename, 'w') as f:
        for subreddit in get_most_active_subreddits(int(sys.argv[2])):
            f.write(subreddit.display_name + '\n')
    
def get_most_active_subreddits(num_subreddits=100):
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent='Research WebScraping')
    subreddits = praw.models.Subreddits(reddit, _data={}).popular(limit=num_subreddits)
    return subreddits

if __name__ == '__main__':
    main()

