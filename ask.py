import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# setting the environment

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db_batch"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.nuwaapi.com/v1"
)

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
# user_query = "Tell me about compost?"
user_query = "Tell me about Fertilizing?"
query_embedding = get_embedding(user_query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=4
)

print(results['documents'])
print(results.keys())
type(results['documents'])
print(results['documents'])
print(results['metadatas'])

# client = OpenAI()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.nuwaapi.com/v1"
)

system_prompt = """
You are a helpful assistant. You answer questions about growing vegetables in Florida. 
But you only answer based on knowledge I'm providing you. You don't use your internal 
knowledge and you don't make thins up.
If you don't know the answer, just say: I don't know
--------------------
The data:
"""+str(results['documents'])+"""
"""

#print(system_prompt)

response = client.chat.completions.create(
    model="gpt-4o",
    messages = [
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_query}    
    ]
)

print("\n\n---------------------\n\n")

print(response.choices[0].message.content)