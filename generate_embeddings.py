from langchain_ollama import OllamaEmbeddings
import pandas as pd

embeddings = OllamaEmbeddings(
    model="llama3.1:8b"
)

def embed_text_in_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    
    # Create a DataFrame to return based on the column name
    if column_name == "comment":
        
        df = df[df['is_conversation'] != 'Bot']
        # Ensure the column data (the actual text) is passed to the embedding function
        text_data = df[column_name].tolist()  # Convert the column to a list of text
        emb = embeddings.embed_documents(text_data)  # Embed the actual text

        return pd.DataFrame({
            'submission_id': df['submission_id'],
            'comment_id': df['comment_id'],
            'comment_emb': emb
        })
    
    elif column_name == "submission_title":

        # Ensure the column data (the actual text) is passed to the embedding function
        text_data = df[column_name].tolist()  # Convert the column to a list of text
        emb = embeddings.embed_documents(text_data)  # Embed the actual text

        return pd.DataFrame({
            'submission_id': df['submission_id'],
            'title_emb': emb
        })
    
    elif column_name == "sub_summary":

        df = df[df['is_conversation'] != 'Bot']
        # Ensure the column data (the actual text) is passed to the embedding function
        text_data = df[column_name].tolist()  # Convert the column to a list of text
        emb = embeddings.embed_documents(text_data)  # Embed the actual text

        return pd.DataFrame({
            'submission_id': df['submission_id'],
            'sub_emb': emb
        })
    else:
        raise ValueError("Invalid column name provided. Must be 'comment', 'submission_title', or 'sub_summary'.")
