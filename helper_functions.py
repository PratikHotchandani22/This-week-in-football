from datetime import datetime, timedelta
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import ast
import asyncio
import ollama
from configurations import BATCH_SIZE
import pandas as pd
import re

async def load_tokenizer_model():
    # Load the tokenizer and model
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    return tokenizer, model

def get_previous_week_range():
    today = datetime.now().date()  # Only keep the date part (no time)
    
    # Find how many days we are from Monday of this week
    days_to_monday = today.weekday() + 1  # weekday() gives 0 for Monday, so +1 gets us to the previous Sunday
    
    # Previous Sunday (end of last week)
    previous_sunday = today - timedelta(days=days_to_monday)
    
    # Previous Monday (start of last week)
    previous_monday = previous_sunday - timedelta(days=6)
    
    # Return the previous Monday and Sunday as a tuple (dates only)
    return previous_monday, previous_sunday

def translate_text_old(input_text, tgt_lang, tokenizer, model):
    # Step 1: Detect the source language
    src_lang = detect(input_text)

    # Step 2: Set the source language for the tokenizer
    tokenizer.src_lang = src_lang

    # Step 3: Encode the input text
    encoded_input = tokenizer(input_text, return_tensors="pt")

    # Step 4: Generate the translated text
    generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))

    # Step 5: Decode the translated text
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return src_lang, translated_text

def translate_text(input_text, tgt_lang, tokenizer, model):
    # Step 1: Detect the source language
    src_lang = detect(input_text)

    if src_lang != 'en':

        # Step 2: Set the source language for the tokenizer
        tokenizer.src_lang = src_lang

        # Step 3: Encode the input text
        encoded_input = tokenizer(input_text, return_tensors="pt")

        # Step 4: Generate the translated text
        generated_tokens = model.generate(**encoded_input, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))

        # Step 5: Decode the translated text
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        return src_lang, translated_text
    
    else:
        return src_lang, "translation not required"

# Function to apply the translation to each comment in the DataFrame
def translate_comments(df, column_name, tgt_lang, tokenizer, model):
    # Initialize new columns for detected language and dynamic translated text
    translated_column_name = f"{column_name}_translated_text"  # Create dynamic column name
    language_column_name = f"{column_name}_langugage"  # Create dynamic column name
    df[language_column_name] = None
    df[translated_column_name] = None

    # Iterate over each row and apply the translation
    for idx in df.index:
        comment = df.at[idx, column_name]  # Access the comment in the given row
        try:

            #TODO: dont translate comments that are not text (could be a link, emojis)
            #TODO: dont translate submissions that are not text
            # Translate the comment
            detected_lang, translated_comment = translate_text(comment, tgt_lang, tokenizer, model)

            # If translation is not required, keep the original comment
            if translated_comment == "translation not required":
                df.at[idx, translated_column_name] = comment
            else:
                df.at[idx, translated_column_name] = translated_comment

            df.at[idx, language_column_name] = detected_lang

        except Exception as e:
            print(f"Error translating comment at index {idx}: {comment}, error: {e}")
            df.at[idx, language_column_name] = None
            df.at[idx, translated_column_name] = None

    return df

def prepare_data_for_translation(df, column_name):
    df['comments'] = df['comments'].apply(lambda x: str(x) if not isinstance(x, str) else x)
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_exploded = df.explode(column_name)
    df_exploded = df_exploded.reset_index(drop=True)

    return df_exploded

def clean_data_after_preparation(df):
    # drop rows where there are no comments.
    df_cleaned = df.dropna(subset=['comments'])

    # Remove rows where the character length of 'comments' is less than or equal to 4
    df_filtered = df_cleaned[df_cleaned['comments'].str.len() >= 3]

    # Remove newline characters from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].replace('\n', ' ', regex=True)

    # Remove rows where comments contain '[deleted]' or '[removed]'
    df_filtered = df_filtered[~df_filtered['comments'].str.contains(r'\[deleted\]|\[removed\]', regex=True)]

    # Remove rows where the comments are only emojis
    emoji_pattern = r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\s]+$'
    df_filtered = df_filtered[~df_filtered['comments'].str.match(emoji_pattern)]

    # Remove rows where the comments are only URLs
    url_pattern = r'^https?://\S+$'
    df_filtered = df_filtered[~df_filtered['comments'].str.match(url_pattern)]

    return df_filtered

async def classify_comments_for_cleaning(prompt, model, df):
    try:
        # Ensure prompt is a non-empty string
        if not isinstance(prompt["content"], str) or not prompt["content"].strip():
            raise ValueError("Prompt must be a non-empty string.")
        else: 
            print("prompt all good")
        
        print("Prompt passed is: ", prompt)

        base_string = prompt["content"]
        responses = []

        # Iterate over DataFrame rows in batches
        for i in range(0, len(df), BATCH_SIZE):
            print(f"Processing comments from row: {i} ...")
            
            for row in df[i:i+BATCH_SIZE].itertuples(index=True):
                prompt = base_string + f"comment: '''{row.comments_translated_text}'''"
                print("Final prompt is: ", prompt)
                print("Generating llama response .... ")

                # Send the custom prompt to the LLaMA 3.1 model
                response = ollama.generate(
                    model=model,
                    prompt=prompt
                )

                # Extract the 'response' from the LLaMA output and store it
                responses.append(response.get('response', ''))

        # Add the LLaMA responses as a new column in the original DataFrame
        df['phi_response'] = pd.Series(responses)

        return df

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except KeyError as ke:
        return f"Response Error: {str(ke)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

async def classify_comments_for_cleaning_new(prompt, model, df):
    try:
        # Ensure prompt is a non-empty string
        if not isinstance(prompt["content"], str) or not prompt["content"].strip():
            raise ValueError("Prompt must be a non-empty string.")
        else:
            print("Prompt all good")
        
        print("Prompt passed is: ", prompt)

        base_string = prompt["content"]
        responses = []

        # Iterate over DataFrame rows in batches
        for i in range(0, len(df), BATCH_SIZE):
            print(f"Processing comments from row: {i} ...")
            
            for row in df[i:i+BATCH_SIZE].itertuples(index=True):
                # Combine the submission title and comment text in the prompt
                full_prompt = base_string + f"submission title: '''{row.submission_title}'''\ncomment: '''{row.comments_translated_text}'''"
                print("Final prompt is: ", full_prompt)
                print("Generating LLaMA response .... ")

                # Send the custom prompt to the LLaMA 3.1 model
                response = ollama.generate(
                    model=model,
                    prompt=full_prompt
                )

                # Extract the 'response' from the LLaMA output and store it
                responses.append(response.get('response', ''))

        # Add the LLaMA responses as a new column in the original DataFrame
        df['llama8b_submission_response'] = pd.Series(responses)

        return df

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except KeyError as ke:
        return f"Response Error: {str(ke)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

def validate_llama_response(df, labels_list):
    """
    Check if the values in the 'llama_response' column are in labels_list.
    Remove blank spaces, quotes, and \n from the responses.
    Replace invalid labels with 'Error'.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the llama responses.
    labels_list (list): A list of valid labels.

    Returns:
    pd.DataFrame: The updated DataFrame with invalid labels replaced by 'Error'.
    """
    # Ensure the labels_list is a set for faster lookup
    valid_labels = set(labels_list)

    # Clean and validate the 'llama_response' column
    df['llama8b_submission_response'] = df['llama8b_submission_response'].apply(
        lambda x: x.replace('"', '').replace("'", "").replace('\n', '').strip() 
        if isinstance(x, str) else x
    )

    # Replace invalid labels with 'Error'
    df['llama8b_submission_response'] = df['llama8b_submission_response'].apply(
        lambda x: x if x in valid_labels else 'Error'
    )

    return df

def remove_bot_NA_error_responses(df):
    df = df[df['llama8b_submission_response'] == 'Human-Conversation']
    return df