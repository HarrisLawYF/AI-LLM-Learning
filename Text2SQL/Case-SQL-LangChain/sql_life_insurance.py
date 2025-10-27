#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase

db_user = "student123"
db_password = "student321"
db_host = "rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306"
db_name = "life_insurance"
db = SQLDatabase.from_uri(f"sqlite:///../identifier.sqlite")
#engine = create_engine('mysql+mysqlconnector://root:passw0rdcc4@localhost:3306/wucai')
db


# In[2]:

from langchain_openai import ChatOpenAI
import os

# 从环境变量获取 dashscope 的 API Key
api_key = os.getenv('DASHSCOPE_API_KEY')

llm = ChatOpenAI(
    temperature=0.01,
    model="qwen-turbo-latest",
    # openai_api_key = "sk-9846f14a2104490b960adbf5c5b3b32e",
    # openai_api_base="https://api.deepseek.com"
    openai_api_base = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    openai_api_key  = api_key
)
# 需要设置llm
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Task1
agent_executor.invoke("Get phone number of all customers")


# In[3]:


agent_executor.invoke("Get all orders of each customer")


# In[4]:


agent_executor.invoke("Get dob of all customers")

