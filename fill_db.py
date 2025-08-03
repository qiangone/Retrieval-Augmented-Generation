from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.nuwaapi.com/v1"
)


DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = chroma_client.get_or_create_collection(name="growing_vegetables")

# loading the document

loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# splitting the document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)
type(chunks)
# preparing to be added in chromadb

documents = []
metadata = []
ids = []
embeddings = []

len(documents)
len(metadata)
len(ids)
len(embeddings)

i = 0

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


for chunk in chunks:
    documents.append(chunk.page_content)
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)
    embeddings.append(get_embedding(chunk.page_content))
    i += 1

  


# adding to chromadb


collection.add(
    documents=documents,
    metadatas=metadata,
    embeddings=embeddings,
    ids=ids
)

