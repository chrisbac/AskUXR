# Import system libraries
import os
import shutil
from dotenv import load_dotenv

# Importing Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

print("=====================================")

# Loading text files
data_folder="./data-input/"
loader = DirectoryLoader(data_folder, glob="*.txt", loader_cls=TextLoader, recursive=True)
documents = loader.load()
print(f">> Loading documents from {data_folder}...")

# Splitting text
chunk_size=600
chunk_overlap=20
print("=====================================")
print(f">> Splitting text in chunks of {chunk_size} tokens, with overlaps of {chunk_overlap}...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
texts = text_splitter.split_documents(documents)

# Printing chunks
for text_chunk in texts:

    # Assuming text_chunk has an attribute 'text' that contains the string
    text_content = text_chunk.text if hasattr(text_chunk, 'text') else str(text_chunk)

    # Calculating the number of characters (or tokens) in the chunk
    text_length = len(text_content)

    # Printing the character (or token) length followed by the text chunk
    print(f"{text_length}: {text_content} \n")
print("=====================================")

# Creating a persisted Vector DB
persist_directory = './data-output/db-chat'
print(f">> Deleting {persist_directory} to avoid conflicts...")
shutil.rmtree(persist_directory)
print(f">> Creating a persisted Chroma vector store in a new {persist_directory}...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
docsearch = Chroma.from_documents(documents=texts,
                                embedding=embeddings,
                                persist_directory=persist_directory)
docsearch.persist()

# Printing final ingestion report
print(">> Done! \n\nFinal report:")
print(f"{len(texts)} chunks")
print(f"{chunk_size} Chunk size (characters)")
print(f"{chunk_overlap} Chunk overlap")
print("=====================================")
