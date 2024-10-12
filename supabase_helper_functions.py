import pandas as pd
from datetime import datetime

def prepare_data_reddit_submission(models_response: pd.DataFrame):
    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame
    for _, row in models_response.iterrows():
        # Prepare the JSON structure for each row
        data = {
            "submission_id": row.get("submission_id", None),
            "submission_title": row.get("submission_title", None),
            "submission_date": row["submission_date"].isoformat() if isinstance(row["submission_date"], pd.Timestamp) else None,  # Ensure it's timezone-aware
            "no_of_upvotes": row.get("no_of_upvotes", None),
            "submission_type": row.get("submission_type", None),
            "submission_url": row.get("submission_url", None),
            "submission_object_url": row.get("submission_object_url", None),
            "comment": row.get("comment", None),  # Assuming this field contains the comment text
            "sentiment": row.get("sentiment", None),  # Assuming sentiment analysis is performed
            "is_conversation": row.get("is_conversation", None),  # Assuming category is assigned
            "subreddit": row.get("subreddit", None),  # Subreddit field
            "links_in_comment": row.get("links_in_comment", None),
            "comment_id": row.get("comment_id", None),
            "comments_upvote": row.get("comments_upvote", None),
            "sub_title_sentiment": row.get("sub_title_sentiment", None),
            "sub_title_emotions": row.get("sub_title_emotions", None),
            "comment_sentiment": row.get("comment_sentiment", None),
            "comment_emotions": row.get("comment_emotions", None)
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries
    return all_data

def prepare_data_reddit_embeddings(df: pd.DataFrame, emb_comment: list, emb_title: list, emb_summary: list):
    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame and corresponding embeddings
    for idx, row in df.iterrows():
        # Prepare the JSON structure for each row, including the embeddings
        data = {
            "submission_id": row.get("submission_id", None),   # Extract submission_id
            "title_emb": emb_title[idx] if idx < len(emb_title) else None,  # Handle embedding for title
            "comment_emb": emb_comment[idx] if idx < len(emb_comment) else None, # Handle embedding for comment
            "sub_emb": emb_summary[idx] if idx < len(emb_summary) else None # Handle embedding for comment
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries (ready for insertion into the database)
    return all_data

def prepare_data_reddit_summary(df: pd.DataFrame):
    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame and corresponding embeddings
    for idx, row_data in df.iterrows():  # Unpack the index and row data
        # Prepare the JSON structure for each row, including the embeddings
        data = {
            "submission_id": row_data.get("submission_id", None),   # Extract submission_id
            "summary_embedding": row_data.get("summary_embedding", None),  # Handle embedding for summary
            "summary": row_data.get("summary", None)   # Extract the summary from the DataFrame
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries (ready for insertion into the database)
    return all_data
