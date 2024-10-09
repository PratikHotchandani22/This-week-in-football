import asyncpraw
import pandas as pd
from datetime import datetime, timezone
from credentials import REDDIT_USER_AGENT, REDDIT_CLIENT_SECRET, REDDIT_CLIENT_ID

async def initialize_reddit_client():
    reddit = asyncpraw.Reddit(
        client_id = REDDIT_CLIENT_ID,
        client_secret = REDDIT_CLIENT_SECRET,
        user_agent = REDDIT_USER_AGENT 
    )
    return reddit

async def get_submission_type(submission):
    submission_vars = vars(submission)  # Get the attributes of the submission

    # Check if 'is_self' attribute is available and determine type
    if 'is_self' in submission_vars and submission.is_self:
        return "Text"
    
    # Check if 'poll_data' attribute is available
    if 'poll_data' in submission_vars and submission.poll_data:
        return "Poll"
    
    # Check if 'url' attribute is available for images
    if 'url' in submission_vars:
        if any(ext in submission.url for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
            return "Image"
        
        # Check for video links
        if any(video_site in submission.url for video_site in ['youtube.com', 'vimeo.com', 'youtu.be']):
            return "Video"

    # If none of the above conditions match, return "Link"
    return "Link"

# Function to scrape data from a list of subreddits within a date range
async def scrape_subreddits(subreddit_list, reddit, start_date, end_date):
    data = []

    # Convert start and end dates to datetime objects if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Loop through each subreddit in the list
    for subreddit_name in subreddit_list:
        subreddit = await reddit.subreddit(subreddit_name)

        # Fetch submissions in the subreddit and filter by date range
        async for submission in subreddit.hot(limit=15):  # Adjust limit as needed
            submission_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).date()

            # Check if submission falls within the date range
            if start_date <= submission_date <= end_date:
                await submission.load()  # Load full submission to access comments

                # Retrieve top 10 comments
                await submission.comments.replace_more(limit=0)  # Expand all comments
                top_comments = submission.comments[:20]  # Get top 100 comments
                comments_text = [comment.body for comment in top_comments] if top_comments else ["No comments"]

                # Identify submission type
                submission_type = await get_submission_type(submission)

                if submission_type == 'Text':
                    submission_object_link = "N/A"
                else:
                    submission_object_link = submission.url

                # Collect submission details
                data.append({
                    'subreddit': subreddit.display_name,
                    'submission_date': submission_date,
                    'submission_id': submission.id,
                    'submission_type': submission_type,
                    'submission_object_link': submission_object_link,
                    'submission_url': submission.permalink,
                    'submission_title': submission.title,
                    'no_of_upvotes': submission.score,
                    'comments': comments_text
                })

    # Convert collected data into a DataFrame
    return pd.DataFrame(data)

def save_scrapped_reddit_data_csvJson(dataframe):
    # Save the DataFrame to a JSON file
    dataframe.to_json('reddit_submissions.json', orient='records', lines=True)

    # Save the DataFrame to a CSV file
    dataframe.to_csv('reddit_submissions.csv', index=False)