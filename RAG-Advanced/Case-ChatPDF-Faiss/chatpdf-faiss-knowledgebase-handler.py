import json
import re

import dashscope
import faiss
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import os
import pickle

from numpy.f2py.auxfuncs import throw_error
from openai import OpenAI

from datetime import datetime
import numpy as np

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=dashscope.api_key,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

TEXT_EMBEDDING_MODEL = "text-embedding-v4"
TEXT_EMBEDDING_DIM = 1024

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

def get_text_embedding(text):
    """获取文本的 Embedding"""
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=text,
        dimensions=TEXT_EMBEDDING_DIM
    )
    return response.data[0].embedding

class KnowledgeBaseVersionManager:
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.versions = {}

    def create_version(self, knowledge_base, version_name, description=""):
        """创建知识库版本"""
        # 构建向量索引
        metadata_store, text_index = self.build_vector_index(knowledge_base)

        version_info = {
            "version_name": version_name,
            "description": description,
            "created_date": datetime.now().isoformat(),
            "knowledge_base": knowledge_base,
            "metadata_store": metadata_store,
            "text_index": text_index,
            "statistics": self.calculate_version_statistics(knowledge_base)
        }

        self.versions[version_name] = version_info
        return version_info

    def build_vector_index(self, knowledge_base):
        """构建向量索引"""
        metadata_store = []
        text_vectors = []

        for i, chunk in enumerate(knowledge_base):
            print(chunk)
            content = chunk.get('content', '')
            if not content.strip():
                continue

            metadata = {
                "id": i,
                "content": content,
                "chunk_id": chunk.get('id', f'chunk_{i}')
            }

            # 获取文本embedding
            vector = get_text_embedding(content)
            text_vectors.append(vector)
            metadata_store.append(metadata)

        # 创建FAISS索引
        text_index = faiss.IndexFlatL2(TEXT_EMBEDDING_DIM)
        text_index_map = faiss.IndexIDMap(text_index)

        if text_vectors:
            text_ids = [m["id"] for m in metadata_store]
            text_index_map.add_with_ids(np.array(text_vectors).astype('float32'), np.array(text_ids))

        return metadata_store, text_index_map

    def calculate_version_statistics(self, knowledge_base):
        """计算版本统计信息"""
        total_chunks = len(knowledge_base)
        total_content_length = sum(len(chunk.get('content', '')) for chunk in knowledge_base)

        return {
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunk_length": total_content_length / total_chunks if total_chunks > 0 else 0
        }

    def compare_versions(self, version1_name, version2_name):
        """比较两个版本的差异"""
        if version1_name not in self.versions or version2_name not in self.versions:
            return {"error": "版本不存在"}

        v1 = self.versions[version1_name]
        v2 = self.versions[version2_name]

        kb1 = v1['knowledge_base']
        kb2 = v2['knowledge_base']

        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "changes": self.detect_changes(kb1, kb2),
            "statistics_comparison": self.compare_statistics(v1['statistics'], v2['statistics'])
        }

        return comparison

    def detect_changes(self, kb1, kb2):
        """检测知识库变化"""
        changes = {
            "added_chunks": [],
            "removed_chunks": [],
            "modified_chunks": [],
            "unchanged_chunks": []
        }

        # 创建ID映射
        kb1_dict = {chunk.get('id'): chunk for chunk in kb1}
        kb2_dict = {chunk.get('id'): chunk for chunk in kb2}

        # 检测新增和删除
        kb1_ids = set(kb1_dict.keys())
        kb2_ids = set(kb2_dict.keys())

        added_ids = kb2_ids - kb1_ids
        removed_ids = kb1_ids - kb2_ids
        common_ids = kb1_ids & kb2_ids

        # 记录新增的知识切片
        for chunk_id in added_ids:
            changes["added_chunks"].append({
                "id": chunk_id,
                "content": kb2_dict[chunk_id].get('content', '')
            })

        # 记录删除的知识切片
        for chunk_id in removed_ids:
            changes["removed_chunks"].append({
                "id": chunk_id,
                "content": kb1_dict[chunk_id].get('content', '')
            })

        # 检测修改的知识切片
        for chunk_id in common_ids:
            chunk1 = kb1_dict[chunk_id]
            chunk2 = kb2_dict[chunk_id]

            if chunk1.get('content') != chunk2.get('content'):
                changes["modified_chunks"].append({
                    "id": chunk_id,
                    "old_content": chunk1.get('content', ''),
                    "new_content": chunk2.get('content', '')
                })
            else:
                changes["unchanged_chunks"].append(chunk_id)

        return changes

    def compare_statistics(self, stats1, stats2):
        """比较统计信息"""
        comparison = {}

        for key in stats1.keys():
            if key in stats2:
                if isinstance(stats1[key], (int, float)):
                    comparison[key] = {
                        "version1": stats1[key],
                        "version2": stats2[key],
                        "difference": stats2[key] - stats1[key],
                        "percentage_change": ((stats2[key] - stats1[key]) / stats1[key] * 100) if stats1[
                                                                                                      key] != 0 else 0
                    }
                elif isinstance(stats1[key], dict):
                    comparison[key] = self.compare_dict_statistics(stats1[key], stats2[key])

        return comparison

    def compare_dict_statistics(self, dict1, dict2):
        """比较字典类型的统计信息"""
        comparison = {}
        all_keys = set(dict1.keys()) | set(dict2.keys())

        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            comparison[key] = {
                "version1": val1,
                "version2": val2,
                "difference": val2 - val1
            }

        return comparison

    def evaluate_version_performance(self, version_name, test_queries):
        """评估版本性能"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}

        performance_metrics = {
            "version_name": version_name,
            "evaluation_date": datetime.now().isoformat(),
            "query_results": [],
            "overall_metrics": {}
        }

        total_queries = len(test_queries)
        correct_answers = 0
        response_times = []

        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')

            # 使用embedding检索
            start_time = datetime.now()
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            end_time = datetime.now()

            response_time = (end_time - start_time).total_seconds()
            response_times.append(response_time)

            # 评估检索质量
            is_correct = self.evaluate_retrieval_quality(query, retrieved_chunks, expected_answer)
            if is_correct:
                correct_answers += 1

            performance_metrics["query_results"].append({
                "query": query,
                "retrieved_chunks": len(retrieved_chunks),
                "response_time": response_time,
                "is_correct": is_correct
            })

        # 计算整体指标
        accuracy = correct_answers / total_queries if total_queries > 0 else 0
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        performance_metrics["overall_metrics"] = {
            "accuracy": accuracy,
            "avg_response_time": avg_response_time,
            "total_queries": total_queries,
            "correct_answers": correct_answers
        }

        return performance_metrics

    def retrieve_relevant_chunks(self, query, version_name, k=3):
        """使用embedding和faiss检索相关知识切片"""
        if version_name not in self.versions:
            return []

        version_info = self.versions[version_name]
        metadata_store = version_info['metadata_store']
        text_index = version_info['text_index']

        # 获取查询的embedding
        query_vector = np.array([get_text_embedding(query)]).astype('float32')

        # 使用faiss进行检索
        distances, indices = text_index.search(query_vector, k)

        relevant_chunks = []
        for i, doc_id in enumerate(indices[0]):
            if doc_id != -1:  # faiss返回-1表示没有找到匹配
                # 通过ID在元数据中查找
                match = next((item for item in metadata_store if item["id"] == doc_id), None)
                if match:
                    # 构造返回的知识切片格式
                    chunk = {
                        "id": match["chunk_id"],
                        "content": match["content"],
                        "similarity_score": 1.0 / (1.0 + distances[0][i])  # 将距离转换为相似度
                    }
                    relevant_chunks.append(chunk)

        return relevant_chunks

    def evaluate_retrieval_quality(self, query, retrieved_chunks, expected_answer):
        """评估检索质量"""
        if not retrieved_chunks:
            return False

        # 简化的质量评估
        for chunk in retrieved_chunks:
            content = chunk.get('content', '').lower()
            if expected_answer.lower() in content:
                return True

        return False

    def compare_version_performance(self, version1_name, version2_name, test_queries):
        """比较两个版本的性能"""
        perf1 = self.evaluate_version_performance(version1_name, test_queries)
        perf2 = self.evaluate_version_performance(version2_name, test_queries)

        if "error" in perf1 or "error" in perf2:
            return {"error": "版本评估失败"}

        comparison = {
            "version1": version1_name,
            "version2": version2_name,
            "comparison_date": datetime.now().isoformat(),
            "performance_comparison": {
                "accuracy": {
                    "version1": perf1["overall_metrics"]["accuracy"],
                    "version2": perf2["overall_metrics"]["accuracy"],
                    "improvement": perf2["overall_metrics"]["accuracy"] - perf1["overall_metrics"]["accuracy"]
                },
                "response_time": {
                    "version1": perf1["overall_metrics"]["avg_response_time"],
                    "version2": perf2["overall_metrics"]["avg_response_time"],
                    "improvement": perf1["overall_metrics"]["avg_response_time"] - perf2["overall_metrics"][
                        "avg_response_time"]
                }
            },
            "recommendation": self.generate_performance_recommendation(perf1, perf2)
        }

        return comparison

    def generate_performance_recommendation(self, perf1, perf2):
        """生成性能建议"""
        acc1 = perf1["overall_metrics"]["accuracy"]
        acc2 = perf2["overall_metrics"]["accuracy"]
        time1 = perf1["overall_metrics"]["avg_response_time"]
        time2 = perf2["overall_metrics"]["avg_response_time"]

        if acc2 > acc1 and time2 <= time1:
            return f"推荐使用版本2，准确率提升{(acc2 - acc1) * 100:.1f}%，响应时间{'提升' if time2 < time1 else '相当'}"
        elif acc2 > acc1 and time2 > time1:
            return f"版本2准确率更高但响应时间较长，需要权衡"
        elif acc2 < acc1 and time2 < time1:
            return f"版本2响应更快但准确率较低，需要权衡"
        else:
            return f"推荐使用版本1，性能更优"

    def generate_regression_test(self, version_name, test_queries):
        """生成回归测试"""
        if version_name not in self.versions:
            return {"error": "版本不存在"}

        regression_results = {
            "version_name": version_name,
            "test_date": datetime.now().isoformat(),
            "test_results": [],
            "pass_rate": 0
        }

        passed_tests = 0
        total_tests = len(test_queries)

        for query_info in test_queries:
            query = query_info['query']
            expected_answer = query_info.get('expected_answer', '')

            # 执行测试
            retrieved_chunks = self.retrieve_relevant_chunks(query, version_name)
            is_passed = self.evaluate_retrieval_quality(query, retrieved_chunks, expected_answer)

            if is_passed:
                passed_tests += 1

            regression_results["test_results"].append({
                "query": query,
                "expected": expected_answer,
                "retrieved": len(retrieved_chunks),
                "passed": is_passed
            })

        regression_results["pass_rate"] = passed_tests / total_tests if total_tests > 0 else 0

        return regression_results

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码

    参数:
        pdf: PDF文件对象

    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))

    return text, page_numbers

def process_text_with_advanced_semantic_chunking_with_llm(text, max_chunk_size=512, save_path: str = None) -> FAISS:
    """使用LLM进行高级语义切片"""
    # 检查环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    prompt = f"""
请将以下文本按照语义完整性进行切片，每个切片不超过{max_chunk_size}字符。
要求：
1. 保持语义完整性
2. 在自然的分割点切分
3. 返回JSON格式的切片列表，格式如下：
{{
  "chunks": [
    "第一个切片内容",
    "第二个切片内容",
    ...
  ]
}}

文本内容：
{text}

请返回JSON格式的切片列表：
"""

    try:
        print("正在调用LLM进行语义切片...")
        response = client.chat.completions.create(
            model="qwen-turbo-latest",
            messages=[
                {"role": "system",
                 "content": "你是一个专业的文本切片助手。请严格按照JSON格式返回结果，不要添加任何额外的标记。"},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        print(f"LLM返回结果: {result[:200]}...")

        # 清理结果，移除可能的Markdown代码块标记
        cleaned_result = result.strip()
        if cleaned_result.startswith('```'):
            # 移除开头的 ```json 或 ```
            cleaned_result = re.sub(r'^```(?:json)?\s*', '', cleaned_result)
        if cleaned_result.endswith('```'):
            # 移除结尾的 ```
            cleaned_result = re.sub(r'\s*```$', '', cleaned_result)

        # 解析JSON结果
        chunks_data = json.loads(cleaned_result)
        chunks = []

        # 处理不同的返回格式
        if "chunks" in chunks_data:
            chunks = chunks_data["chunks"]
        else:
            print(f"意外的返回格式: {chunks_data}")
            throw_error("No chunks data")

        # 创建嵌入模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )

        # 从文本块创建知识库
        knowledgeBase = FAISS.from_texts(chunks, embeddings)
        print("已从文本块创建知识库。")

        # 为每个chunk找到最匹配的页码
        page_info = {}
        for i, chunk in enumerate(chunks):
            page_info[chunk] = i

        knowledgeBase.page_info = page_info

        # 如果提供了保存路径，则保存向量数据库和页码信息
        if save_path:
            # 确保目录存在
            os.makedirs(save_path, exist_ok=True)

            # 保存FAISS向量数据库
            knowledgeBase.save_local(save_path)
            print(f"向量数据库已保存到: {save_path}")

            # 保存页码信息到同一目录
            with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
                pickle.dump(page_info, f)
            print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

        return knowledgeBase

    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}")
        print(f"原始结果: {result}")
        pass
    except Exception as e:
        print(f"LLM切片失败: {e}")

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 改进：存储每个文本块对应的页码信息
    # 创建原始文本的行列表和对应的页码列表
    lines = text.split("\n")
    
    # 为每个chunk找到最匹配的页码
    page_info = {}
    for chunk in chunks:
        # 查找chunk在原始文本中的开始位置
        start_idx = text.find(chunk[:100])  # 使用chunk的前100个字符作为定位点
        if start_idx == -1:
            # 如果找不到精确匹配，则使用模糊匹配
            for i, line in enumerate(lines):
                if chunk.startswith(line[:min(50, len(line))]):
                    start_idx = i
                    break
            
            # 如果仍然找不到，尝试另一种匹配方式
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break
        
        # 如果找到了起始位置，确定对应的页码
        if start_idx != -1:
            # 计算这个位置对应原文中的哪一行
            line_count = text[:start_idx].count("\n")
            # 确保不超出页码列表长度
            if line_count < len(page_numbers):
                page_info[chunk] = page_numbers[line_count]
            else:
                # 如果超出范围，使用最后一个页码
                page_info[chunk] = page_numbers[-1] if page_numbers else 1
        else:
            # 如果无法匹配，使用默认页码-1（这里应该根据实际情况设置一个合理的默认值）
            page_info[chunk] = -1
    
    knowledgeBase.page_info = page_info
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码信息到同一目录
        with open(os.path.join(save_path, "page_info.pkl"), "wb") as f:
            pickle.dump(page_info, f)
        print(f"页码信息已保存到: {os.path.join(save_path, 'page_info.pkl')}")

    return knowledgeBase

def extract_data_from_knowledege_base(knowledgeBase):
    texts = knowledgeBase.docstore._dict
    data = []
    for doc_id, doc in texts.items():
        item = {  # create new dict each time
            "id": doc_id,
            "content": doc.page_content
        }
        data.append(item)
    return data


def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码信息
    page_info_path = os.path.join(load_path, "page_info.pkl")
    if os.path.exists(page_info_path):
        with open(page_info_path, "rb") as f:
            page_info = pickle.load(f)
        knowledgeBase.page_info = page_info
        print("页码信息已加载。")
    else:
        print("警告: 未找到页码信息文件。")
    
    return knowledgeBase

# 读取PDF文件
pdf_reader = PdfReader('./test.pdf')
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_numbers(pdf_reader)
text


print(f"提取的文本长度: {len(text)} 个字符。")


# 处理文本并创建知识库，同时保存到磁盘
save_dir = "./vector_db_v1"
knowledge_base_v1_wrapper = process_text_with_splitter(text, page_numbers, save_path=save_dir)
knowledge_base_v1 = extract_data_from_knowledege_base(knowledge_base_v1_wrapper)
save_dir = "./vector_db_v2"
knowledge_base_v2_wrapper = process_text_with_advanced_semantic_chunking_with_llm(text, max_chunk_size=2000, save_path=save_dir)
knowledge_base_v2 = extract_data_from_knowledege_base(knowledge_base_v2_wrapper)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
# 从磁盘加载向量数据库
knowledge_base_v1_wrapper = load_knowledge_base("./vector_db_v1", embeddings)
knowledge_base_v2_wrapper = load_knowledge_base("./vector_db_v2", embeddings)
knowledge_base_v1 = extract_data_from_knowledege_base(knowledge_base_v1_wrapper)
knowledge_base_v2 = extract_data_from_knowledege_base(knowledge_base_v2_wrapper)

version_manager = KnowledgeBaseVersionManager()
print("功能1: 创建知识库版本")
v1_info = version_manager.create_version(knowledge_base_v1, "v1.0", "基础版本")
v2_info = version_manager.create_version(knowledge_base_v2, "v2.0", "增强版本")

print(f"版本1信息:")
print(f"  版本名: {v1_info['version_name']}")
print(f"  描述: {v1_info['description']}")
print(f"  知识切片数量: {v1_info['statistics']['total_chunks']}")
print(f"  平均切片长度: {v1_info['statistics']['average_chunk_length']:.0f}字符")

print(f"\n版本2信息:")
print(f"  版本名: {v2_info['version_name']}")
print(f"  描述: {v2_info['description']}")
print(f"  知识切片数量: {v2_info['statistics']['total_chunks']}")
print(f"  平均切片长度: {v2_info['statistics']['average_chunk_length']:.0f}字符")

print("\n" + "=" * 60 + "\n")

# 功能示例2: 版本比较
print("功能2: 版本差异比较")
comparison = version_manager.compare_versions("v1.0", "v2.0")

print(f"版本比较结果:")
changes = comparison['changes']
print(f"  新增知识切片: {len(changes['added_chunks'])}个")
print(f"  删除知识切片: {len(changes['removed_chunks'])}个")
print(f"  修改知识切片: {len(changes['modified_chunks'])}个")

print(f"\n新增的知识切片:")
for i, chunk in enumerate(changes['added_chunks'], 1):
    print(f"  {i}. ID: {chunk['id']}")
    print(f"     内容: {chunk['content']}")

print(f"\n修改的知识切片:")
for i, chunk in enumerate(changes['modified_chunks'], 1):
    print(f"  {i}. ID: {chunk['id']}")
    print(f"     旧内容: {chunk['old_content']}")
    print(f"     新内容: {chunk['new_content']}")

print("\n" + "=" * 60 + "\n")

# 功能3: 性能评估
print("功能3: 版本性能评估")

test_queries = [
    {"query": "客户经理被投诉了，投诉一次扣多少分？", "expected_answer": "2分"},  # 关键词包含即正确
    {"query": "工作责任心不强，投诉一次扣多少分？", "expected_answer": "5分"},
    {"query": "未按规定要求进行贷前调查、贷后检查工作的，投诉一次扣多少分？", "expected_answer": "5分"},
    {"query": "客户经理准入条件的工作条件？", "expected_answer": "工作经历：须具备大专以上学历，至少二年以上银行工作经验"}
]

perf_v1 = version_manager.evaluate_version_performance("v1.0", test_queries)
perf_v2 = version_manager.evaluate_version_performance("v2.0", test_queries)

print(f"版本1性能:")
print(f"  准确率: {perf_v1['overall_metrics']['accuracy'] * 100:.1f}%")
print(f"  平均响应时间: {perf_v1['overall_metrics']['avg_response_time'] * 1000:.1f}ms")

print(f"\n版本2性能:")
print(f"  准确率: {perf_v2['overall_metrics']['accuracy'] * 100:.1f}%")
print(f"  平均响应时间: {perf_v2['overall_metrics']['avg_response_time'] * 1000:.1f}ms")

print("\n" + "=" * 60 + "\n")

# 功能4: 性能比较
print("功能4: 性能比较与建议")
perf_comparison = version_manager.compare_version_performance("v1.0", "v2.0", test_queries)

print(f"性能比较结果:")
comp = perf_comparison['performance_comparison']
print(f"  准确率提升: {comp['accuracy']['improvement'] * 100:.1f}%")
print(f"  响应时间变化: {comp['response_time']['improvement'] * 1000:.1f}ms")
print(f"  建议: {perf_comparison['recommendation']}")

print("\n" + "=" * 60 + "\n")

# 功能5: 回归测试
print("功能5: 回归测试")
regression_v2 = version_manager.generate_regression_test("v2.0", test_queries)

print(f"回归测试结果:")
print(f"  测试通过率: {regression_v2['pass_rate'] * 100:.1f}%")
print(f"  测试用例数量: {len(regression_v2['test_results'])}")

print(f"\n详细测试结果:")
for i, result in enumerate(regression_v2['test_results'], 1):
    status = "✓" if result['passed'] else "✗"
    print(f"  {i}. {result['query']} {status}")