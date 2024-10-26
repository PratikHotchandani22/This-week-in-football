import pandas as pd
from datetime import datetime
import ast
import json
import numpy as np

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

def safe_eval(x):
    if isinstance(x, str):  # Only apply literal_eval if the value is a string
        try:
            return tuple(ast.literal_eval(x))
        except (ValueError, SyntaxError):  # Catch exceptions if literal_eval fails
            return ()
    elif isinstance(x, (list, tuple, np.ndarray)):  # If it's already a list, tuple, or array, return as a tuple
        return tuple(x)
    else:
        return ()

def prepare_data_reddit_embeddings(emb_comment_df: pd.DataFrame, emb_title_df: pd.DataFrame, emb_summary_df: pd.DataFrame):
    
 # Apply the safe_eval function to 'title_emb', 'sub_emb', and 'comment_emb'
    emb_title_df['title_emb'] = emb_title_df['title_emb'].apply(safe_eval)
    emb_summary_df['sub_emb'] = emb_summary_df['sub_emb'].apply(safe_eval)
    emb_comment_df['comment_emb'] = emb_comment_df['comment_emb'].apply(safe_eval)
    
    # Drop duplicates (tuples are hashable, so this works)
    emb_title_df = emb_title_df.drop_duplicates()
    emb_summary_df = emb_summary_df.drop_duplicates()

    # Merge the DataFrames on the 'submission_id' column
    merged_df = pd.merge(emb_title_df, emb_summary_df, on='submission_id', how='left')
    merged_df = pd.merge(merged_df, emb_comment_df, on='submission_id', how='left')

    # Convert back to lists after removing duplicates
    merged_df['title_emb'] = merged_df['title_emb'].apply(lambda x: list(x))
    merged_df['sub_emb'] = merged_df['sub_emb'].apply(lambda x: list(x) if pd.notnull(x) else [])
    merged_df['comment_emb'] = merged_df['comment_emb'].apply(lambda x: list(x) if pd.notnull(x) else [])

    # Replace NaN values with None for any other fields if necessary
    merged_df = merged_df.where(pd.notnull(merged_df), None)

    all_data = merged_df.to_dict(orient='records')

    return merged_df, all_data

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

def prepare_data_reddit_weekly_summary(df: pd.DataFrame):
    # Initialize a list to hold all the prepared data for each row
    all_data = []

    # Iterate over each row in the DataFrame and corresponding embeddings
    for idx, row_data in df.iterrows():  # Unpack the index and row data
        # Prepare the JSON structure for each row, including the embeddings
        data = {
            "week_start_date": row_data["week_start_date"].isoformat() if isinstance(row_data["week_start_date"], pd.Timestamp) else None,
            "week_end_date": row_data["week_end_date"].isoformat() if isinstance(row_data["week_end_date"], pd.Timestamp) else None,  # Ensure it's timezone-aware
            "weekly_summary": row_data.get("weekly_summary", None),   # Extract the summary from the DataFrame
            "weekly_summary_embedding": row_data.get("weekly_summary_embedding", None),   # Extract the summary from the DataFrame
        }

        # Append the prepared data for this row to the list
        all_data.append(data)

    # Return the list of dictionaries (ready for insertion into the database)
    return all_data

def prepare_data_league_matches(matches_df: pd.DataFrame):
    all_data = []
    
    for _, row in matches_df.iterrows():
        data = {
            "round": row["round"] if pd.notna(row["round"]) else None,
            "wk": row["wk"] if pd.notna(row["wk"]) else None,
            "day": row["day"] if pd.notna(row["day"]) else None,
            "match_date": row["match_date"].isoformat() if isinstance(row["match_date"], pd.Timestamp) else None,
            "home": row["home"] if pd.notna(row["home"]) else None,
            "xg_home": float(row["xg_home"]) if pd.notna(row["xg_home"]) else None,  # Numeric (4,2) field
            "score": row["score"] if pd.notna(row["score"]) else None,
            "xg_away": float(row["xg_away"]) if pd.notna(row["xg_away"]) else None,  # Numeric (4,2) field
            "away": row["away"] if pd.notna(row["away"]) else None,
            "attendance": int(row["attendance"]) if pd.notna(row["attendance"]) else None,  # Integer field
            "venue": row["venue"] if pd.notna(row["venue"]) else None,
            "referee": row["referee"] if pd.notna(row["referee"]) else None,
            "league_name": row["league_name"] if pd.notna(row["league_name"]) else None,
            "unique_identifier_id": row["unique_identifier_id"] if pd.notna(row["unique_identifier_id"]) else None
        }

        all_data.append(data)

    # Save all_data to a JSON file for inspection
    with open("league_matches_data.json", "w") as json_file:
        json.dump(all_data, json_file, indent=4)

    return all_data
