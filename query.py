# coding=utf-8
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# 加载环境变量，确保 .env 文件中的 OPENAI_API_KEY 可用
load_dotenv()

# --- 配置部分 ---
# PDF 文件所在的目录
DATA_PATH = "data"
# ChromaDB 数据库存储路径
CHROMA_PATH = "chroma_db_batch"
# 集合名称，你可以自定义
COLLECTION_NAME = "growing_vegetables"
# 嵌入模型名称
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- 简化后的查询函数 ---

def query_collection(query_text: str):
    """
    对 ChromaDB 集合进行相似性查询。
    """
    print("\n--- 启动文档查询流程 ---")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        # 获取已有的集合，它已经绑定了 embedding_function
        collection = chroma_client.get_collection(name=COLLECTION_NAME,embedding_function=embedding_function)
        print(f"成功连接到集合 '{COLLECTION_NAME}'，其中包含 {collection.count()} 个文档。")
    except Exception as e:
        print(f"获取集合失败，请确保你已经运行过摄取脚本。错误: {e}")
        return

    # 1. 直接传入字符串进行查询
    # ChromaDB 会自动使用绑定的 embedding_function 将其转换为嵌入向量
    print(f"正在执行相似性查询...")
    results = collection.query(
        query_texts=[query_text],  # <--- 将 query_embeddings 替换为 query_texts
        n_results=3,
        include=['documents', 'metadatas', 'distances']
    )

    # 2. 打印查询结果（这部分逻辑不变）
    print("\n--- 查询结果 ---")
    if results['documents']:
        for i, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
            print(f"结果 {i+1}:")
            print(f"  相似度（距离）: {distance:.4f}")
            print(f"  来源文件: {metadata.get('source', '未知')}")
            print(f"  内容片段: {doc[:200]}...")
            print("-" * 20)
    else:
        print("未找到相关结果。")



try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.nuwaapi.com/v1"
    )
except Exception as e:
    print(f"初始化 OpenAI 客户端失败: {e}")
    exit()

# 确保 embedding_function 使用的是正确的模型
embedding_function = OpenAIEmbeddingFunction(api_key=client.api_key, model_name=EMBEDDING_MODEL)

# user_query = "Tell me about compost?"
user_query = "Tell me about Fertilizing?"
query_collection(user_query)