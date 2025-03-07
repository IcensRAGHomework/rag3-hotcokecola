import datetime
import chromadb
import traceback
import csv

from chromadb.utils import embedding_functions
from model_configurations import get_model_configuration
from datetime import datetime


gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_file = 'COA_OpenData.csv'

def generate_hw01():

    # Create embedding function   
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=dbpath)

    # Create a new collection
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    ) 
    
    # Function to parse dates and calculate seconds from the epoch start date
    def calculate_unix_timestamp(date_str):
        # Parse the given date
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # Return the Unix timestamp
        return int(date.timestamp())

    if collection.count() == 0:
        with open(csv_file, mode='r', encoding='utf-8-sig') as file:
            csv_reader = csv.DictReader(file) 
        
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
    #get collection from hw1
    collection = generate_hw01()
    
    results = collection.query(
        query_texts = question,
        n_results = 10,
        where={"$and":
                [{'city' : {"$in": city}}, 
                {'type' : {"$in": store_type}},
                {'date' : {"$gte": int(start_date.timestamp())}},
                {'date' : {"$lte": int(end_date.timestamp())}}]
            }
    )

    #print(results)


    # Filter names based on the distance threshold   
    threshold = 0.8
    distances = results['distances'][0]
    distances = [1 - distance for distance in distances]
    metadatas = results['metadatas'][0]
    #print(distances)
    #print(metadatas)

    filtered_names = [metadata['name'] for distance, metadata in zip(distances, metadatas) if distance > threshold]
    #print(filtered_names)

    return filtered_names
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    
    #get collection from hw1
    collection = generate_hw01()
    
    #find store_name
    results = collection.query(
        query_texts = [store_name],
        n_results = 1
    )

    #print(results)
    #print(results['ids'][0][0])
    
    collection.update(
        ids = results['ids'][0][0],  # Assuming the ID of the item to update is '1'
        metadatas = {'name': new_store_name}
    )

    #query user question
    results = collection.query(
        query_texts = question,
        n_results = 10,
        where={"$and":
                [{'city' : {"$in": city}}, 
                {'type' : {"$in": store_type}}]
            }
    )

    # Filter names based on the distance threshold   
    threshold = 0.8
    distances = results['distances'][0]
    distances = [1 - distance for distance in distances]
    metadatas = results['metadatas'][0]
    #print(distances)
    #print(metadatas) 
    filtered_names = [metadata['name'] for distance, metadata in zip(distances, metadatas) if distance > threshold]
    #print(filtered_names)
    
    return filtered_names
    
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
