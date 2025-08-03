from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings

loader = PyPDFLoader("data/VH021.pdf")
docs = loader.load_and_split()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.8)

embeddings = OpenAIEmbeddings()

chroma_db = Chroma.from_documents(
    documents=docs, 
    embedding=embeddings, 
    persist_directory="data", 
    collection_name="lc_chroma_demo"
)