from langchain_ollama import OllamaEmbeddings
import pandas as pd

embeddings = OllamaEmbeddings(
    model="llama3.1:8b"
)

def embed_text_in_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    def get_valid_text_data(df, column_name):
        # Filter out rows where 'is_conversation' is 'Bot' if applicable
        if 'is_conversation' in df.columns:
            df = df[df['is_conversation'] != 'Bot']
        
        # Ensure only strings are passed and are non-empty
        valid_rows = df[column_name].apply(lambda x: isinstance(x, str) and bool(x))
        df_valid = df[valid_rows]
        
        text_data = df_valid[column_name].tolist()
        
        # Check if text_data is not empty
        if not text_data:
            raise ValueError(f"No valid text data in column {column_name}")
        
        return df_valid, text_data
    
    # Handle embeddings based on the column name
    if column_name == "comment":
        df_valid, text_data = get_valid_text_data(df, column_name)
        emb = embeddings.embed_documents(text_data)

        return pd.DataFrame({
            'submission_id': df_valid['submission_id'].values,
            'comment_id': df_valid['comment_id'].values,
            'comment_emb': emb
        })

    elif column_name == "submission_title":
        df_valid, text_data = get_valid_text_data(df, column_name)
        emb = embeddings.embed_documents(text_data)

        return pd.DataFrame({
            'submission_id': df_valid['submission_id'].values,
            'title_emb': emb
        })

    elif column_name == "sub_summary":
        df_valid, text_data = get_valid_text_data(df, column_name)
        emb = embeddings.embed_documents(text_data)

        return pd.DataFrame({
            'submission_id': df_valid['submission_id'].values,
            'sub_emb': emb
        })

    elif column_name == "llm_summary_response":
        df_valid, text_data = get_valid_text_data(df, column_name)
        emb = embeddings.embed_documents(text_data)

        return pd.DataFrame({
            'week_start_date': df_valid['week_start_date'].values,
            'week_end_date': df_valid['week_end_date'].values,
            'weekly_summary': df_valid['llm_summary_response'].values,
            'weekly_summary_embedding': emb
        })

    else:
        raise ValueError("Invalid column name provided. Must be 'comment', 'submission_title', 'sub_summary', or 'llm_summary_response'.")
