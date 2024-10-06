from configurations import REDDIT_SUBREDDITS, tgt_lang, PROMPT_COMMENT_CLEANING, REDDIT_COMMENT_CLEANING_LABELS_STR, MODEL_COMMENT_CLEANING, MODEL_COMMENT_CLEANING_2
from helper_functions import get_previous_week_range, load_tokenizer_model, translate_comments, prepare_data_for_translation, clean_data_after_preparation, classify_comments_for_cleaning
from scrapping_reddit import initialize_reddit_client, scrape_subreddits, save_scrapped_reddit_data_csvJson
import asyncio
import pandas as pd
import ast


async def main():
    
    """
    start_date, end_date = get_previous_week_range()
    reddit = await initialize_reddit_client()
    print("initialized reddit client: ", reddit)
    scrapped_data = await scrape_subreddits(REDDIT_SUBREDDITS, reddit, start_date, end_date)
    print("scrapped data is: ", scrapped_data)
    await reddit.close()
    """

    """
    df = pd.read_csv("/Users/pratikhotchandani/Downloads/Github/This-week-in-football/reddit_submissions.csv")
    df = df[df['subreddit'] == 'Bundesliga'].sample(10)
    df = prepare_data_for_translation(df, 'comments')
    prepared_df = clean_data_after_preparation(df)
    

    #loading translation model and tokenizer
    tokenizer, model = await load_tokenizer_model()
    translated_data = translate_comments(prepared_df, 'comments', 'en', tokenizer, model)

    # Create a new DataFrame where each submission_id is unique
    unique_submission_df = translated_data.drop_duplicates(subset='submission_id', keep='first')
    translated_data_title = translate_comments(unique_submission_df, 'submission_title', 'en', tokenizer, model)

    # Ensure both dataframes have the required columns
    translated_data_title_reduced = translated_data_title[['submission_id', 'submission_title_translated_text']]

    # Merge df1 with df2 on submission_id
    merged_df = translated_data.merge(translated_data_title_reduced, on='submission_id', how='left')

    # Replace the values in submission_title column in df1 with the translated title from df2
    translated_data['submission_title'] = merged_df['submission_title_translated_text'].combine_first(translated_data['submission_title'])
    translated_data.to_csv("translations.csv")

    # Display the resulting DataFrame
    print(translated_data)
    """
    df = pd.read_csv("/Users/pratikhotchandani/Downloads/Github/This-week-in-football/translations.csv")
    print("data read: ", df)
    llama_response = await classify_comments_for_cleaning(PROMPT_COMMENT_CLEANING, "phi3.5", df)
    print("Classification done....")
    #print(llama_response)
    llama_response.to_csv("llama_response.csv")

    """
    comments = scrapped_data[['comments']]
    text = comments.iloc[171].comments

    # Convert the string to a list
    sentences_list = ast.literal_eval(text)
    # Define the target language
    target_language = 'en'  # Change this to your desired target language

    # Prepare a list to hold the results
    results = []

    # Iterate over the sentences and translate each one
    for text in sentences_list:
        detected_lang, translated_text = translate_text(text, target_language, tokenizer, model)
        results.append({
            'original_text': text,
            'detected_language': detected_lang,
            'final_translated_text': translated_text
        })

    # Create a DataFrame from the results
    translations_df = pd.DataFrame(results)
    translations_df.to_csv("translations.csv")
    # Display the resulting DataFrame
    print(translations_df)

    


    print("** Saving scrapped data **")
    save_scrapped_reddit_data_csvJson(scrapped_data)
    
    """





# Ensure the event loop is run properly
if __name__ == "__main__":
    asyncio.run(main())  # Run the async main function