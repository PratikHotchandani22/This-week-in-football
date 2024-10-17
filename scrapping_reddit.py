import asyncpraw
import pandas as pd
from datetime import datetime, timezone
from credentials import REDDIT_USER_AGENT, REDDIT_CLIENT_SECRET, REDDIT_CLIENT_ID
import asyncio

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
        await asyncio.sleep(2)  # Adds a 2-second delay
        
        # Initialize the pagination variable
        after = None

        # Loop to fetch all submissions using pagination
        while True:
            # Fetch submissions in batches of 100 using 'after' for pagination
            submissions = subreddit.hot(limit=100, params={"after": after})
            
            # Stop if no more submissions
            fetched_submissions = 0
            async for submission in submissions:
                fetched_submissions += 1
                submission_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).date()

                # Check if submission falls within the date range
                if start_date <= submission_date <= end_date:
                    await submission.load()  # Load full submission to access comments

                    # Retrieve all comments
                    await submission.comments.replace_more(limit=None)  # Expand all comments
                    await asyncio.sleep(3)  # Adds a 2-second delay
                    top_comments = submission.comments  # Get all comments
                    comments_text = [comment.body for comment in top_comments] if top_comments else ["No comments"]
                    comments_id = [comment.id for comment in top_comments] if top_comments else ["No comments"]
                    comments_upvote = [comment.score for comment in top_comments] if top_comments else ["No comments"]

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
                        'comments': comments_text,
                        'comment_id': comments_id,
                        'comments_upvote': comments_upvote
                    })

            # If fewer than 100 submissions were fetched, stop the loop (end of subreddit)
            if fetched_submissions < 100:
                break

            # Get the last submission's ID to continue fetching the next page
            after = submission.name

    # Convert collected data into a DataFrame
    return pd.DataFrame(data)

def save_scrapped_reddit_data_csvJson(dataframe):
    # Save the DataFrame to a JSON file
    dataframe.to_json('reddit_submissions.json', orient='records', lines=True)

    # Save the DataFrame to a CSV file
    dataframe.to_csv('reddit_submissions.csv', index=False)