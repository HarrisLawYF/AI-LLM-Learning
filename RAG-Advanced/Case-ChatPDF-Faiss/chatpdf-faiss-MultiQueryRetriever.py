from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MultiQueryRetriever
from langchain_community.llms import Tongyi
from typing import List, Tuple
import os
import pickle
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

# 获取环境变量中的 DASHSCOPE_API_KEY
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
if not DASHSCOPE_API_KEY:
    raise ValueError("请设置环境变量 DASHSCOPE_API_KEY")

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
        else:
            print(f"No text found on page {page_number}.")

    return text, page_numbers

def process_text_with_splitter(text: str, page_numbers: List[int], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 改进：存储每个文本块对应的页码信息
    lines = text.split("\n")
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
            if start_idx == -1:
                for i, line in enumerate(lines):
                    if line and line in chunk:
                        start_idx = text.find(line)
                        break
        if start_idx != -1:
            line_count = text[:start_idx].count("\n")
            if line_count < len(page_numbers):
                page_info[chunk] = page_numbers[line_count]
            else:
                page_info[chunk] = page_numbers[-1] if page_numbers else 1
        else:
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
            model="text-embedding-v1",
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

def create_multi_query_retriever(vectorstore, llm):
    """
    创建MultiQueryRetriever
    
    参数:
        vectorstore: 向量数据库
        llm: 大语言模型，用于查询改写
    
    返回:
        retriever: MultiQueryRetriever对象
    """
    # 创建基础检索器
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # 创建MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    return retriever

def process_query_with_multi_retriever(query: str, retriever, llm):
    """
    使用MultiQueryRetriever处理查询
    
    参数:
        query: 用户查询
        retriever: MultiQueryRetriever对象
        llm: 大语言模型
    
    返回:
        response: 回答
        unique_pages: 相关文档的页码集合
    """
    # 执行查询，获取相关文档
    docs = retriever.invoke(query)
    print(f"找到 {len(docs)} 个相关文档")
    
    # 加载问答链
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # 准备输入数据
    input_data = {"input_documents": docs, "question": query}
    
    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = chain.invoke(input=input_data)
        print(f"查询已处理。成本: {cost}")
    
    # 记录唯一的页码
    unique_pages = set()
    
    # 获取每个文档块的来源页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        # 获取向量存储中的页码信息
        source_page = retriever.retriever.vectorstore.page_info.get(
            text_content.strip(), "未知"
        )
        
        if source_page not in unique_pages:
            unique_pages.add(source_page)
    
    return response, unique_pages

def main():
    # 设置PDF文件路径
    pdf_path = './浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf'
    # 设置向量数据库保存路径
    vector_db_path = './vector_db'
    
    # 检查向量数据库是否已存在
    if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
        print(f"发现现有向量数据库: {vector_db_path}")
        # 创建嵌入模型
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
        # 加载向量数据库
        knowledgeBase = load_knowledge_base(vector_db_path, embeddings)
    else:
        print(f"未找到向量数据库，将从PDF创建新的向量数据库")
        # 读取PDF文件
        pdf_reader = PdfReader(pdf_path)
        # 提取文本和页码信息
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        print(f"提取的文本长度: {len(text)} 个字符。")
        
        # 处理文本并创建知识库，同时保存到磁盘
        knowledgeBase = process_text_with_splitter(text, page_numbers, save_path=vector_db_path)
    
    # 初始化大语言模型（用于查询改写和回答生成）
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 创建MultiQueryRetriever
    multi_retriever = create_multi_query_retriever(knowledgeBase, llm)
    
    # 设置查询问题
    queries = [
        "客户经理被投诉了，投诉一次扣多少分",
        "客户经理每年评聘申报时间是怎样的？",
        "客户经理的考核标准是什么？"
    ]
    
    # 处理每个查询
    for query in queries:
        print("\n" + "="*50)
        print(f"查询: {query}")
        
        # 使用MultiQueryRetriever处理查询
        response, unique_pages = process_query_with_multi_retriever(
            query, 
            multi_retriever, 
            llm
        )
        
        # 打印回答
        print("\n回答:")
        print(response["output_text"])
        
        # 打印来源页码
        print("\n来源页码:")
        for page in sorted(unique_pages):
            print(f"- 第 {page} 页")
        print("="*50)

if __name__ == "__main__":
    main()

