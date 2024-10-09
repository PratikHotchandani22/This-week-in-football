from configurations import REDDIT_SUBREDDITS, tgt_lang, REDDIT_COMMENT_CLEANING_LABELS, PROMPT_COMMENT_CLEANING_SUBMISSION, REDDIT_COMMENT_CLEANING_LABELS_STR, MODEL_COMMENT_CLEANING_2
from helper_functions import get_previous_week_range, load_tokenizer_model, translate_comments, prepare_data_for_translation, clean_data_after_preparation, classify_comments_for_cleaning, classify_comments_for_cleaning_new, remove_bot_NA_error_responses, validate_llama_response
from scrapping_reddit import initialize_reddit_client, scrape_subreddits, save_scrapped_reddit_data_csvJson
import asyncio
import pandas as pd
import ast

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
    llama_response = await classify_comments_for_cleaning_new(PROMPT_COMMENT_CLEANING_SUBMISSION, "phi3.5:latest", translated_data)
    print("Classification done....")

    llama_response.to_csv("llama8b_submission_new.csv")

    print("cleaning llama responses....")
    validated_response = validate_llama_response(llama_response, REDDIT_COMMENT_CLEANING_LABELS)
    #filtered_response = remove_bot_NA_error_responses(validated_response)

    print("saving final responseeee")
    validated_response.to_csv("final_response_new.csv")

    print("task done...")
    """
    comments = scrapped_data[['comments']]
    text = comments.iloc[171].comments

    # Convert the string to a list
    sentences_list = ast.literal_eval(text)
    """

# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function