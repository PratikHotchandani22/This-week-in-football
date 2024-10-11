from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="llama3.1:8b"
)

def embed_text_in_column(df, column_name):
    # Ensure the column data (the actual text) is passed to the embedding function, not the column name string
    text_data = df[column_name].tolist()  # Convert the column to a list of text
    emb = embeddings.embed_documents(text_data)  # Embed the actual text
    return emb
    