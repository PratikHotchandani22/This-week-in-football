from datetime import datetime, timedelta
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langchain_ollama import ChatOllama
import ast
import asyncio
import ollama
from configurations import BATCH_SIZE
import pandas as pd
import re
import spacy
import emoji
import time

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

def translate_text(input_text, tgt_lang, tokenizer, model):
    # Step 1: Detect the source language
    src_lang = detect(input_text)

    # TODO: Dont translate if the text is just numbers

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
    # Ensure all columns with lists (comments, comment_id, comments_upvote) are evaluated as lists
    df['comments'] = df['comments'].apply(lambda x: ast.literal_eval(x) if not isinstance(x, list) else x)
    df['comment_id'] = df['comment_id'].apply(lambda x: ast.literal_eval(x) if not isinstance(x, list) else x)
    df['comments_upvote'] = df['comments_upvote'].apply(lambda x: ast.literal_eval(x) if not isinstance(x, list) else x)

    # Explode the comments, comment_id, and comments_upvote columns simultaneously
    df_exploded = df.explode(['comments', 'comment_id', 'comments_upvote'])

    # Reset index to clean up the exploded DataFrame
    df_exploded = df_exploded.reset_index(drop=True)

    return df_exploded

# Function to extract URLs from the text and remove them from the original comments
def extract_links(comment):
    # Regex pattern to match URLs
    url_pattern = r'(https?://[^\s]+)'
    
    # Find all URLs in the comment
    urls = re.findall(url_pattern, comment)
    
    # Remove URLs from the comment
    comment_without_urls = re.sub(url_pattern, '', comment).strip()
    
    # If any URLs were found, return them as a string, else return None
    return comment_without_urls, ', '.join(urls) if urls else None

# Function to remove emojis from the text
def remove_emojis(text):
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)

def check_valid_sentence(comment):

    # Load the English language model
    nlp = spacy.load("en_core_web_sm")

    # Process the comment with spaCy
    doc = nlp(comment)
    
    # Initialize counters for alphabetic characters and check if it's all numeric
    alphabetic_char_count = 0
    all_numeric = True
    
    for token in doc:
        # Count each alphabetic character, not just full tokens
        alphabetic_char_count += sum(1 for char in token.text if char.isalpha())
        
        # Check if the token contains any non-numeric characters
        if not token.is_digit:
            all_numeric = False
    
    # If the text is all numeric, return it as is
    if all_numeric:
        return comment
    
    # Check if there are at least 3 alphabetic characters (instead of tokens)
    if alphabetic_char_count >= 3:
        return comment  # Return the original comment if it's valid
    
    # Otherwise, return an empty string
    return ''  # Return an empty string if it doesn't meet the criteria

def clean_data_after_preparation(df):

    # drop rows where there are no comments.
    df_cleaned = df.dropna(subset=['comments'])

    df_cleaned['original_comments'] = df_cleaned['comments']

    # Remove rows where the character length of 'comments' is less than or equal to 4
    df_filtered = df_cleaned[df_cleaned['comments'].str.len() >= 3]

    # Remove newline characters from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].replace('\n', ' ', regex=True)

    # Remove rows where comments contain '[deleted]' or '[removed]'
    df_filtered = df_filtered[~df_filtered['comments'].str.contains(r'\[deleted\]|\[removed\]', regex=True)]

    # Remove quotes from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('"', '', regex=False)
    
    # Remove asterisks from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('*', '', regex=False)

    # Remove square brackets from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('[', '', regex=False).str.replace(']', '', regex=False)

    # Remove curly brackets from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('(', '', regex=False).str.replace(')', '', regex=False)

    # Remove "√º" from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('√', '', regex=False)

    # Remove hyphens from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('-', '', regex=False)

    # Remove tildes from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('~', '', regex=False)

    # Remove dots from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('.', '', regex=False)

    # Remove tildes from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].str.replace('>', '', regex=False)

    # Apply the function to the 'comments' column and create a new 'links_in_comment' column
    df_filtered[['comments', 'links_in_comment']] = df_filtered['comments'].apply(lambda x: pd.Series(extract_links(x)))

    # Apply the function to remove emojis from the 'comments' column
    df_filtered['comments'] = df_filtered['comments'].apply(remove_emojis)

    # Apply the check_valid_sentence function to the 'comments' column
    df_filtered['val_comments'] = df_filtered['comments'].apply(check_valid_sentence)
    
    return df_filtered

async def classify_comments_for_cleaning_new(prompt, model, df):
    try:
        # Ensure prompt is a non-empty string
        if not isinstance(prompt["content"], str) or not prompt["content"].strip():
            raise ValueError("Prompt must be a non-empty string.")
        else:
            print("Prompt all good")
        base_string = prompt["content"]
        responses = []

        # Iterate over DataFrame rows in batches
        for i in range(0, len(df), BATCH_SIZE):
            print(f"Processing comments from row: {i} ...")
            
            for row in df[i:i+BATCH_SIZE].itertuples(index=True):
                # Combine the submission title and comment text in the prompt
                full_prompt = base_string + f"submission title: '''{row.submission_title}'''\ncomment: '''{row.val_comments_translated_text}'''"
                print("Final prompt is: ", full_prompt)
                print("Generating LLaMA response .... ")

                # Send the custom prompt to the LLaMA 3.1 model
                response = ollama.generate(
                    model=model,
                    prompt=full_prompt
                )
                print("comment passed: ", row.val_comments_translated_text)
                print("the response that we got is: ", response.get('response', ''))

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

async def classify_comments_for_cleaning_with_models(prompt, models, df):
    try:
         # Ensure prompt is a non-empty string
        if not isinstance(prompt["content"], str) or not prompt["content"].strip():
            raise ValueError("Prompt must be a non-empty string.")
        else:
            print("Prompt all good")

        base_string = prompt["content"]

        # Iterate over the list of models
        for model in models:
            responses = []

            # Iterate over DataFrame rows in batches
            for i in range(0, len(df), BATCH_SIZE):
                print(f"Processing comments from row: {i} with model: {model} ...")

                for row in df[i:i+BATCH_SIZE].itertuples(index=True):
                    full_prompt = base_string + f"submission title: '''{row.submission_title}'''\ncomment: '''{row.val_comments_translated_text}'''"
                    # Send the custom prompt to the model
                    response = ollama.generate(
                        model=model,
                        prompt=full_prompt
                    )

                    # Extract the 'response' from the model output and store it
                    responses.append(response.get('response', ''))

            # Add the model responses as a new column in the original DataFrame
            # Column name is based on the model
            df[f"{model}_response"] = pd.Series(responses)

        return df

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except KeyError as ke:
        return f"Response Error: {str(ke)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

async def classify_comments_for_cleaning_with_models_time(prompt, models, df):
    try:
        # Ensure prompt is a non-empty string
        if not isinstance(prompt["content"], str) or not prompt["content"].strip():
            raise ValueError("Prompt must be a non-empty string.")
        else:
            print("Prompt all good")

        base_string = prompt["content"]

        # Dictionary to store total execution times for each model
        model_execution_times = {}

        # Iterate over the list of models
        for model in models:
            responses = []

            # Start the timer for the current model
            start_time = time.time()

            # Iterate over DataFrame rows in batches
            for i in range(0, len(df), BATCH_SIZE):
                print(f"Processing comments from row: {i} with model: {model} ...")

                for row in df[i:i + BATCH_SIZE].itertuples(index=True):
                    full_prompt = base_string + f"submission title: '''{row.submission_title}'''\ncomment: '''{row.val_comments_translated_text}'''"
                    # Send the custom prompt to the model
                    response = ollama.generate(
                        model=model,
                        prompt=full_prompt
                    )

                    # Extract the 'response' from the model output and store it
                    responses.append(response.get('response', ''))

            # Add the model responses as a new column in the original DataFrame
            df[f"{model}_response"] = pd.Series(responses)

            # Calculate total execution time for this model
            end_time = time.time()
            total_time = end_time - start_time

            # Store execution time for the model
            model_execution_times[model] = total_time

        # Convert the model execution times into a DataFrame
        time_df = pd.DataFrame(list(model_execution_times.items()), columns=['Model', 'Execution Time (seconds)'])

        return df, time_df

    except ValueError as ve:
        return f"Input Error: {str(ve)}"
    except KeyError as ke:
        return f"Response Error: {str(ke)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

# Define the async function to classify comments
async def classify_comments_for_cleaning_with_models_time_langChain(prompt, model_name, df, batch_size=100):
    
    responses = []  # Store model responses

    try:
        print("Prompt template is valid.")

        # Start the timer for the model
        start_time = time.time()

        # Define the LLM with customizable parameters
        llm = ChatOllama(
            model=model_name,
            temperature=0.2  # Adjust temperature as needed
        )

        # Iterate over DataFrame rows in batches
        for i in range(0, len(df), batch_size):
            print(f"Processing comments from row: {i} with model: {llm.model} ...")

            # Process each batch
            for row in df[i:i + batch_size].itertuples(index=True):
                # Prepare the messages for the LLM
                messages = [
                    ("system", prompt),
                    ("human", f"Submission title: {row.submission_title}\n```{row.val_comments_translated_text}```")
                ]

                # Invoke the model
                response = llm.invoke(messages)

                # Append the response
                responses.append(response.content)

        # Add the responses as a new column in the original DataFrame
        df[f"{llm.model}_response"] = pd.Series(responses)


    except Exception as e:
        print(f"Error occurred: {e}")
        return df, pd.DataFrame()  # Return empty DataFrame for time_df on error

    return df

def validate_qwen_response(df, labels_list):
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
    df['qwen2.5:7b_response'] = df['qwen2.5:7b_response'].apply(
        lambda x: x.replace('"', '').replace("'", "").replace('\n', '').strip() 
        if isinstance(x, str) else x
    )

    # Replace invalid labels with 'Error'
    df['qwen2.5:7b_response'] = df['qwen2.5:7b_response'].apply(
        lambda x: x if x in valid_labels else 'Error'
    )

    df = df[df['qwen2.5:7b_response'] != ""]

    # drop rows where there are no comments.
    df = df.dropna(subset=['qwen2.5:7b_response'])

    # Remove newline characters from the 'comments' column
    df['qwen2.5:7b_response'] = df['qwen2.5:7b_response'].replace('\n', ' ', regex=True)

    return df

def clean_data_for_sentiment(df):

    drop_columns = ['comments', 'original_comments','val_comments','val_comments_langugage']
    
    # Assuming df is your DataFrame
    #df = df.drop(df.columns[0], axis=1)
    df = df.drop(columns=drop_columns)
    df['submission_date'] = pd.to_datetime(df['submission_date'])

    df = df.rename(columns={'val_comments_translated_text': 'comment', 
                            'qwen2.5:7b_response':'is_conversation',
                            'submission_object_link':'submission_object_url'})
    
    return df