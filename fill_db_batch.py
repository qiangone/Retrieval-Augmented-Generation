# coding=utf-8
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

# 初始化 OpenAI 客户端
try:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.nuwaapi.com/v1"
    )
except Exception as e:
    print(f"初始化 OpenAI 客户端失败: {e}")
    exit()

def get_embedding_in_batches(texts: List[str], model: str) -> List[List[float]]:
    """
    通过批量调用 OpenAI API 获取文本嵌入向量，提高效率。
    """
    embeddings = []
    # OpenAI 官方文档建议批量大小为 2048，这里我们使用一个更保守的值以确保兼容性
    batch_size = 100
    print(f"正在分批生成嵌入向量 (每批次 {batch_size} 个)...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        try:
            # 核心：一次 API 调用处理一批文本
            response = client.embeddings.create(input=batch_texts, model=model)
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            print(f"已完成批次 {i // batch_size + 1} 的嵌入向量生成。")
        except Exception as e:
            print(f"生成嵌入向量时发生错误 (批次 {i // batch_size + 1}): {e}")
            break # 发生错误时停止处理

    return embeddings

def main():
    """主函数，负责文档加载、分割和数据摄取。"""
    print("--- 启动文档摄取流程 ---")

    # 初始化 ChromaDB 客户端
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # 检查集合是否已存在
    collection_names = [col.name for col in chroma_client.list_collections()]
    if COLLECTION_NAME in collection_names:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        # 如果集合中有数据，则跳过摄取，防止重复
        if collection.count() > 0:
            print(f"集合 '{COLLECTION_NAME}' 中已包含 {collection.count()} 个文档。跳过数据摄取。")
            print("如果需要重新摄取，请手动删除 'chroma_db' 文件夹并重新运行。")
            return
    else:
        # 如果集合不存在，则创建它
        print(f"集合 '{COLLECTION_NAME}' 不存在，正在创建...")
        # 注意: get_or_create_collection 会自动处理嵌入函数的持久化
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        embedding_function = OpenAIEmbeddingFunction(api_key=client.api_key, model_name=EMBEDDING_MODEL)
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    
    # 1. 加载文档
    print(f"正在从目录 '{DATA_PATH}' 加载文档...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()
    if not raw_documents:
        print("未找到任何 PDF 文档。请确保 'data' 文件夹中存在 PDF 文件。")
        return
    print(f"成功加载 {len(raw_documents)} 个原始文档。")

    # 2. 分割文档
    print("正在分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"文档已被分割为 {len(chunks)} 个块。")

    # 3. 准备数据并生成嵌入向量
    documents = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    # 批量生成嵌入向量，这是提高效率的关键步骤
    embeddings = get_embedding_in_batches(documents, EMBEDDING_MODEL)
    
    if not embeddings:
        print("由于 API 错误，无法生成嵌入向量。数据摄取失败。")
        return

    # 4. 添加到 ChromaDB
    print("正在将文档和嵌入向量添加到 ChromaDB...")
    try:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        print(f"成功将 {collection.count()} 个文档添加到集合 '{COLLECTION_NAME}' 中。")
    except Exception as e:
        print(f"添加文档到 ChromaDB 失败: {e}")

if __name__ == "__main__":
    main()
