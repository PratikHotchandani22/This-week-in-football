from configurations import MODEL_EMOTION_TAGGING, PROMPT_SENTIMENT_EMOTION, REDDIT_COMMENT_CLEANING_LABELS, REDDIT_EMBEDDINGS_TABLE, REDDIT_SUMMARY_TABLE, REDDIT_SUBMISSIONS_TABLE, MODEL_COMMENT_CLEANING, REDDIT_SUBREDDITS, tgt_lang, PROMPT_COMMENT_CLEANING_SUBMISSION, PROMPT_COMMENT_CLEANING_SUBMISSION
from helper_functions import extract_sentiment_emotion, sentiment_emotion_tagging_comments_langchain, clean_data_for_sentiment, validate_qwen_response, get_previous_week_range, classify_comments_for_cleaning_with_models_time_langChain, load_tokenizer_model, translate_comments, prepare_data_for_translation, clean_data_after_preparation
from scrapping_reddit import initialize_reddit_client, scrape_subreddits, save_scrapped_reddit_data_csvJson
import asyncio
import pandas as pd

from generate_embeddings import embed_text_in_column
from supabase_backend import create_supabase_connection, insert_data_into_table, fetch_data_from_table
from supabase_helper_functions import prepare_data_reddit_embeddings, prepare_data_reddit_submission

async def main():
    
    print("Starting llm agent...")
    start_date, end_date = get_previous_week_range()
    reddit = await initialize_reddit_client()
    print("initialized reddit client: ", reddit)

    print("starting llm scraping...")
    scrapped_data = await scrape_subreddits(REDDIT_SUBREDDITS, reddit, start_date, end_date)
    print("scrapped data is: ", scrapped_data)
    await reddit.close()
    
    print("saving scrapped data...")
    save_scrapped_reddit_data_csvJson(scrapped_data)
    
    print("Starting llm translation...")
    #scrapped_data = pd.read_csv("scrapped_data_new.csv")
    prepared_df = prepare_data_for_translation(scrapped_data, 'comments')
    prepared_df = clean_data_after_preparation(prepared_df)
    
    #loading translation model and tokenizer
    tokenizer, model = await load_tokenizer_model()
    
    # translat comments
    translated_data = translate_comments(prepared_df, 'val_comments', tgt_lang, tokenizer, model)

    # Create a new DataFrame where each submission_id is unique
    unique_submission_df = translated_data.drop_duplicates(subset='submission_id', keep='first')
    
    # translate submission_title
    translated_data_title = translate_comments(unique_submission_df, 'submission_title', tgt_lang, tokenizer, model)

    # Ensure both dataframes have the required columns
    translated_data_title_reduced = translated_data_title[['submission_id', 'submission_title_translated_text']]

    # Merge df1 with df2 on submission_id
    merged_df = translated_data.merge(translated_data_title_reduced, on='submission_id', how='left')

    # Replace the values in submission_title column in df1 with the translated title from df2
    translated_data['submission_title'] = merged_df['submission_title_translated_text'].combine_first(translated_data['submission_title'])
    translated_data.to_csv("new_translations.csv")
    print("translation comepleteddd....")

    print("starting llama text comment classification...")
    models_response = await classify_comments_for_cleaning_with_models_time_langChain(PROMPT_COMMENT_CLEANING_SUBMISSION, MODEL_COMMENT_CLEANING, translated_data)
    print("Classification done....")

    models_response.to_csv("models_response.csv")

    print("Cleaning model response for human/bot classification..")
    validated_model_response = validate_qwen_response(models_response, REDDIT_COMMENT_CLEANING_LABELS)
    
    # Assuming df is your DataFrame
    print("Columns at validation: ", validated_model_response.columns.tolist())
    cleaned_df = clean_data_for_sentiment(validated_model_response)
    cleaned_df.to_csv("cleaned.csv")
    
    print("performing sentiment and emotion tagging using llama3.1.....")
    df_emotions = await sentiment_emotion_tagging_comments_langchain(PROMPT_SENTIMENT_EMOTION, MODEL_EMOTION_TAGGING, cleaned_df)
    df_emotions.to_csv("emotion_tagging.csv")

    print("cleaning response after emotion tagging...")
    cleaned_df_emotions = extract_sentiment_emotion(df_emotions, "llama3.1:8b_sentiment_emotion_response")
    print("saving cleaned emotions df...")
    cleaned_df_emotions.to_csv("cleaned_df_emotions.csv")

    reddit_submission_prepared_data = prepare_data_reddit_submission(cleaned_df_emotions)
    print("saving prepared reddit data for supabase...")
    
    print("Creating supbase connection..")
    supabase_client = await create_supabase_connection()
    print("Supabase connection created.. ")

    print("Adding reddit submissions data to supabase table..")
    reddit_supabase_sub_response = insert_data_into_table(supabase_client, REDDIT_SUBMISSIONS_TABLE, reddit_submission_prepared_data)
    print("Supabase submissions response ok")

    print("generating embeddings...")
    emb_comment = embed_text_in_column(cleaned_df,'comment')
    emb_title = embed_text_in_column(cleaned_df,'submission_title')
    print("embedding generated...")
    
    print("structuring data for embeddings table...")
    reddit_embedding_prepared_data = prepare_data_reddit_embeddings(cleaned_df,emb_comment, emb_title, ["",""])

    print("adding data to table in supabase...")
    reddit_supabase_emb_response = insert_data_into_table(supabase_client, REDDIT_EMBEDDINGS_TABLE ,reddit_embedding_prepared_data)
    print("Supabase submissions response ok")
    

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function