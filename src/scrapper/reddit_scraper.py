import os
import praw
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
username = os.getenv("REDDIT_USERNAME")
password = os.getenv("REDDIT_PASSWORD")
user_agent = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    username=username,
    password=password,
    user_agent=user_agent
)

subreddit_categories = {
    "depression": ["r/depression"],
    "anxiety": ["r/Anxiety", "r/socialanxiety", "r/healthanxiety"],
    "ocd": ["r/OCD"],
    "adhd": ["r/ADHD"],
    "bipolar": ["r/BipolarReddit"],
    "addiction": ["r/addiction", "r/alcoholism"],
    "autism": ["r/autism"],
    "bpd": ["r/bpd"],
    "psychosis": ["r/schizophrenia", "r/hallucinations", "r/psychosis"],
    "ptsd": ["r/ptsd"],
    "suicide": ["r/SuicideWatch"]
}

def scrape_subreddit(subreddit_name, limit=1800):
    print(f"Scraping {subreddit_name}...")
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for submission in subreddit.hot(limit=limit):
        posts.append({
            "id": submission.id,
            "title": submission.title,
            "text": submission.selftext,
            "created_utc": submission.created_utc,
            "score": submission.score,
            "num_comments": submission.num_comments,
            "subreddit": submission.subreddit.display_name
        })

    return pd.DataFrame(posts)

def scrape_and_merge_category(category_name, subreddit_list, limit_per_sub=1800):
    all_posts = []
    
    for subreddit_name in subreddit_list:
        clean_name = subreddit_name.replace("r/", "").lower()
        try:
            df = scrape_subreddit(clean_name, limit=limit_per_sub)
            df['category'] = category_name
            all_posts.append(df)
            print(f"Successfully scraped {len(df)} posts from {subreddit_name}")
        except Exception as e:
            print(f"Error scraping {subreddit_name}: {e}")
    
    if all_posts:
        merged_df = pd.concat(all_posts, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['id'], keep='first')
        output_path = f"../../data/{category_name}.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"Saved {len(merged_df)} unique posts to {output_path} for category: {category_name}")
        
        return merged_df
    else:
        print(f"No data collected for category: {category_name}")
        return pd.DataFrame()

if __name__ == "__main__":
    for category, subreddits in subreddit_categories.items():
        try:
            scrape_and_merge_category(category, subreddits, limit_per_sub=1800)
            print(f"Completed category: {category}")
            print("-" * 50)
        except Exception as e:
            print(f"Error processing category {category}: {e}")