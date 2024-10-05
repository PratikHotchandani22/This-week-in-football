from datetime import datetime, timedelta
from langdetect import detect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import ast

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

# Function to apply the translation to each comment in the list
def translate_comments_old(df, column_name, tgt_lang, tokenizer, model):
    # List to hold the results
    translated_results = []

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        comments = row[column_name]  # Get the list of comments in the current row

        # Convert the string to a list
        sentences_list = ast.literal_eval(comments)

        # Check if the value is a list
        if isinstance(sentences_list, list):
            print("first comment row is: ", sentences_list)
            translated_row = []
            for comment in sentences_list:
                try:
                    # Translate each comment
                    detected_lang, translated_comment = translate_text(comment, tgt_lang, tokenizer, model)
                    translated_row.append({
                        'original_text': comment,
                        'detected_language': detected_lang,
                        'translated_text': translated_comment
                    })
                except Exception as e:
                    print(f"Error in row {idx}, comment: {comment}, error: {e}")
                    continue

            # Append translated result for the current row
            translated_results.append(translated_row)

        else:
            print("not an instance of list...")
    
    return translated_results

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
    df[column_name] = df[column_name].apply(ast.literal_eval)
    df_exploded = df.explode(column_name)
    df_exploded = df_exploded.reset_index(drop=True)

    return df_exploded

def clean_data_after_preparation(df):
    df_cleaned = df.dropna(subset=['comments'])
    
    # Remove rows where the character length of 'comments' is less than or equal to 4
    df_filtered = df_cleaned[df_cleaned['comments'].str.len() >= 3]

    return df_filtered


