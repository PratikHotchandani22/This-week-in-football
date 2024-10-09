
async def scrape_subreddit(subreddit_name, reddit):

    # List to store data for DataFrame
    data = []
    subreddit = await reddit.subreddit(subreddit_name)

    # Fetch top 100 submissions from the subreddit
    async for submission in subreddit.hot(limit=1):
        # Load the full submission to access its comments
        await submission.load()

        # Get submission details
        submission_id = submission.id
        submission_title = submission.title
        submission_score = submission.score
        submission_date = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).date()
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
            'submisson_utc': submission_date,
            'submission_title': submission_title,
            'submission_score': submission_score,
            'comments': comments_text
        })

    # Convert data to a DataFrame
    return pd.DataFrame(data)
