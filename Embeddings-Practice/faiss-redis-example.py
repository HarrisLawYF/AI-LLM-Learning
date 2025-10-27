import json
import os
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Tuple, Any
from PyPDF2 import PdfReader
import re
import redis

# Step1. 初始化 API 客户端
try:
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
except Exception as e:
    print("初始化OpenAI客户端失败，请检查环境变量'DASHSCOPE_API_KEY'是否已设置。")
    print(f"错误信息: {e}")
    exit()

r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

def extract_text_with_page_numbers(pdf) -> List[Any]:
    """
    从PDF中提取文本并记录每行文本对应的页码

    参数:
        pdf: PDF文件对象

    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    documents = []
    doc_info = pdf.metadata
    print(doc_info)
    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        metadata = {}
        metadata["Author"] = doc_info["/Author"]
        metadata["Creator"] = doc_info["/Creator"]
        metadata["CreationDate"] = doc_info["/CreationDate"]
        metadata["ModDate"] = doc_info["/ModDate"]
        metadata["Producer"] = doc_info["/Producer"]
        filtered_text = extracted_text.replace("百度文库  - 好好学习，天天向上","")
        filtered_text = filtered_text.replace("\n-1 上海浦东发展银行西安分行","")
        filtered_text = filtered_text.replace("\n个金客户经理管理考核暂行办法  \n \n","")
        matches = re.findall(r"第[一二三四五六七八九十百千]+章\s*[^\n]+", filtered_text)
        if matches:
            for m in matches:
                label = m
                print(label)
                metadata["title"] = label
                if extracted_text:
                    documents.append({"id":page_number, "text":extracted_text, "metadata":metadata})
    return documents

# 读取PDF文件
pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
# Step2. 准备示例文本和元数据
# 在实际应用中，这些数据可能来自数据库、文件等
documents = extract_text_with_page_numbers(pdf_reader)
print(documents)
# Step3. 创建元数据存储和向量列表
# 我们使用一个简单的列表来存储元数据。列表的索引将作为FAISS的ID。
# 这种方式简单直接，适用于中小型数据集。
# 对于大型数据集，可以考虑使用字典或数据库（如Redis, SQLite）
metadata_store = []
vectors_list = []
vector_ids = []

print("正在为文档生成向量...")
for i, doc in enumerate(documents):
    try:
        # 调用API生成向量
        completion = client.embeddings.create(
            model="text-embedding-v4",
            input=doc["text"],
            dimensions=2048,
            encoding_format="float"
        )

        # 获取向量
        vector = completion.data[0].embedding
        vectors_list.append(vector)

        # 存储元数据，并使用列表索引作为唯一ID
        #metadata_store.append(doc)
        vector_ids.append(i)  # 自定义ID与列表索引一致

        # 存储元数据，并使用列表索引作为唯一ID
        r.set(f"doc:{i}", json.dumps(doc))
        print(f"  - 已处理文档 {i + 1}/{len(documents)}")

    except Exception as e:
        print(f"处理文档 '{doc['id']}' 时出错: {e}")
        continue

# 将向量列表转换为NumPy数组，FAISS需要这种格式
vectors_np = np.array(vectors_list).astype('float32')
vector_ids_np = np.array(vector_ids)

# Step4. 构建并填充 FAISS 索引
dimension = 2048  # 向量维度
k = 3  # 查找最近的3个邻居

# 创建一个基础的L2距离索引
index_flat_l2 = faiss.IndexFlatL2(dimension)

# 使用IndexIDMap来包装基础索引，能够映射我们自定义的ID
# 这就是关联向量和元数据的关键！
index = faiss.IndexIDMap(index_flat_l2)

# 将向量和它们对应的ID添加到索引中
index.add_with_ids(vectors_np, vector_ids_np)

print(f"\nFAISS 索引已成功创建，共包含 {index.ntotal} 个向量。")

# Step5. 执行搜索并检索元数据
query_text = "我想了解一下成为客户经理的条件"
print(f"\n正在为查询文本生成向量: '{query_text}'")

try:
    # 为查询文本生成向量
    query_completion = client.embeddings.create(
        model="text-embedding-v4",
        input=query_text,
        dimensions=2048,
        encoding_format="float"
    )
    query_vector = np.array([query_completion.data[0].embedding]).astype('float32')

    # 在FAISS索引中执行搜索
    # search方法返回两个NumPy数组：
    # D: 距离 (distances)
    # I: 索引/ID (indices/IDs)
    distances, retrieved_ids = index.search(query_vector, k)

    # Step6. 展示结果
    print("\n--- 搜索结果 ---")
    # `retrieved_ids[0]` 包含与查询最相似的k个向量的ID
    for i in range(k):
        doc_id = retrieved_ids[0][i]

        # 检查ID是否有效
        if doc_id == -1:
            print(f"\n排名 {i + 1}: 未找到更多结果。")
            continue

        # 使用ID从我们的元数据存储中检索信息
        #retrieved_doc = metadata_store[doc_id]
        retrieved_doc_json = r.get(f"doc:{doc_id}")
        if retrieved_doc_json:
            retrieved_doc = json.loads(retrieved_doc_json)
        else:
            print(f"找不到ID {doc_id} 对应的文档元数据。")
            continue
        print(f"\n--- 排名 {i + 1} (相似度得分/距离: {distances[0][i]:.4f}) ---")
        print(f"ID: {doc_id}")
        print(f"原始文本: {retrieved_doc['text']}")
        print(f"元数据: {retrieved_doc['metadata']}")

except Exception as e:
    print(f"执行搜索时发生错误: {e}")
