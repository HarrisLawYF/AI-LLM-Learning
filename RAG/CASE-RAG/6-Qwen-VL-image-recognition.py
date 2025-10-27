import json
import os
import dashscope
from dashscope.api_entities.dashscope_response import Role
from dashscope import MultiModalConversation
import os
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

local_file_path = 'file://2-万圣节.jpeg'
messages = [{
    'role': 'system',
    'content': [{
        'text': 'You are a helpful assistant.'
    }]
}, {
    'role':
    'user',
    'content': [
        {
            'image': local_file_path
        },
        {
            'text': '这是一张什么海报？'
        },
    ]
}]
response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
print(response)