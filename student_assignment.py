import datetime
import chromadb
import traceback

from chromadb.utils import embedding_functions
from model_configurations import get_model_configuration


import csv
from datetime import datetime


gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():   
    # Initialize ChromaDB client
    client = chromadb.Client()
    
    # Create a new collection
    collection = client.create_collection(name="TRAVEL", metadata={
        "hnsw:space": "cosine"})

    # Function to parse dates and calculate seconds from the epoch start date
    def calculate_unix_timestamp(date_str):
        # Parse the given date
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # Return the Unix timestamp
        return int(date.timestamp())

    # Read CSV file into a DataFrame
    csv_file_path = 'COA_OpenData.csv'

    with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.DictReader(file)
        headers = csv_reader.fieldnames
        #print("CSV Headers:", headers)  # Print headers to verify   

        for row in csv_reader:
            # Add metadata to the collection
            collection.add(
                ids=row['ID'],
                documents=row['HostWords'],
                metadatas={
                    'name': row['Name'],
                    'type': row['Type'],
                    'address': row['Address'],
                    'tel': row['Tel'],
                    'city': row['City'],
                    'town': row['Town'],
                    'date': calculate_unix_timestamp(row['CreateDate'])
                }
            )

    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    pass
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection
