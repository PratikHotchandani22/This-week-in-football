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
            "links_in_comment": row.get("links_in_comment", None) 
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries
    return all_data
