import asyncpraw
import pandas as pd
from credentials import REDDIT_USER_AGENT, REDDIT_CLIENT_SECRET, REDDIT_CLIENT_ID

async def initialize_reddit_client():
    reddit = asyncpraw.Reddit(
        client_id = REDDIT_CLIENT_ID,
        client_secret = REDDIT_CLIENT_SECRET,
        user_agent = REDDIT_USER_AGENT 
    )
    return reddit

async def scrape_subreddit(subreddit_name, reddit):

    # List to store data for DataFrame
    data = []
    subreddit = await reddit.subreddit(subreddit_name)

    # Fetch top 100 submissions from the subreddit
    async for submission in subreddit.hot(limit=100):
        # Load the full submission to access its comments
        await submission.load()

        # Get submission details
        submission_id = submission.id
        submission_title = submission.title
        submission_score = submission.score
        subreddit = submission.subreddit.display_name

        # Retrieve top 10 comments (ensure comments are sorted by 'best' or 'top')
        await submission.comments.replace_more(limit=0)  # Expand all comments
        top_comments = submission.comments[:10]  # Get top 10 comments

        # Collect data for the submission
        comments_text = [comment.body for comment in top_comments] if top_comments else ["No comments"]

        # Add a dictionary to the data list
        data.append({
            'subreddit': subreddit,
            'submission_id': submission_id,
            'submission_title': submission_title,
            'submission_score': submission_score,
            'comments': comments_text
        })

    # Convert data to a DataFrame
    return pd.DataFrame(data)

def save_scrapped_reddit_data_csvJson(dataframe):
    # Save the DataFrame to a JSON file
    dataframe.to_json('reddit_submissions.json', orient='records', lines=True)

    # Save the DataFrame to a CSV file
    dataframe.to_csv('reddit_submissions.csv', index=False)
