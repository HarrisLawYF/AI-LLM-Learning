import re
from typing import List, Union
import textwrap
import time

from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM

# 输出结果显示，每行最多60字符，每个字符显示停留0.1秒（动态显示效果）
def output_response(response: str) -> None:
    """动态显示响应结果"""
    if not response:
        exit(0)
    # 每行最多60个字符
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # 每个字符之间延迟0.1秒
            print(" ", end="", flush=True)  # 单词之间添加空格
        print()  # 每行结束后换行
    # 遇到这里，这个问题的回答就结束了
    print("----------------------------------------------------------------")

# 定义了LLM的Prompt Template
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 模拟公司产品和公司介绍的数据源
class TeslaDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    # 工具1：产品描述
    def find_product_description(self, product_name: str) -> str:
        """模拟公司产品的数据库"""
        product_info = {
            "Model 3": "具有简洁、动感的外观设计，流线型车身和现代化前脸。定价23.19-33.19万",
            "Model Y": "在外观上与Model 3相似，但采用了更高的车身和更大的后备箱空间。定价26.39-36.39万",
            "Model X": "拥有独特的翅子门设计和更加大胆的外观风格。定价89.89-105.89万",
        }
        # 基于产品名称 => 产品描述
        return product_info.get(product_name, "没有找到这个产品")

    # 工具2：公司介绍
    def find_company_info(self, query: str) -> str:
        """模拟公司介绍文档数据库，让llm根据信息回答问题"""
        context = """
        特斯拉最知名的产品是电动汽车，其中包括Model S、Model 3、Model X和Model Y等多款车型。
        特斯拉以其技术创新、高性能和领先的自动驾驶技术而闻名。公司不断推动自动驾驶技术的研发，并在车辆中引入了各种驾驶辅助功能，如自动紧急制动、自适应巡航控制和车道保持辅助等。
        """
        # prompt模板 = 上下文context + 用户的query
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        # 使用LLM进行推理
        return self.llm(prompt)

# 设置通义千问API密钥
DASHSCOPE_API_KEY = 'sk-882e296067b744289acf27e6e20f3ec0'

if __name__ == "__main__":
    # 定义LLM
    llm = Tongyi(model_name="qwen-turbo", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 自有数据源
    tesla_data_source = TeslaDataSource(llm)
    
    # 定义工具
    tools = [
        Tool(
            name="查询产品名称",
            func=tesla_data_source.find_product_description,
            description="通过产品名称找到产品描述时用的工具，输入的是产品名称",
        ),
        Tool(
            name="公司相关信息",
            func=tesla_data_source.find_company_info,
            description="当用户询问公司相关的问题，可以通过这个工具了解公司信息",
        ),
    ]
    
    # 使用ReAct模式初始化Agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )

    # 主过程：可以一直提问下去，直到Ctrl+C
    while True:
        try:
            user_input = input("请输入您的问题：")
            response = agent.run(user_input)
            output_response(response)
        except KeyboardInterrupt:
            print("\n程序已退出")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue
