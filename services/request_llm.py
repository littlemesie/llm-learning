# -*- coding:utf-8 -*-

"""
@date: 2025/4/14 下午2:05
@summary:
"""
import time
import json
import requests
headers = {
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Accept": "/"
}

def request_llm():
    template = f"请对以下内容意图识别，意图为[其他，重复警情, 警情画像, 电话分析]，直接输出意图名称，结构为数组。\n内容：xxx"
    json_data = {
        "model": "deepseek-r1:1.5b",
        "messages": [
            {"role": "user", "content": template}
        ],
        "temperature": 0,
        "max_tokens": 10000,
        "stream": True,
        "stop": ["\n```"]
    }

    url = "http://10.105.11.247:800/v1/chat/completions"
    res = requests.post(url=url, json=json_data, headers=headers, stream=True)
    outputs = ""
    for line in res.iter_lines():
        line = line.decode('utf-8')
        if line:
            try:
                line = line.replace('data: ', '')
                # print(line)
                line = json.loads(line)
                delta = line['choices'][0]['delta']
                if 'content' not in delta:
                    break
                outputs = outputs + delta['content']
            except json.decoder.JSONDecodeError:
                break
    print(outputs)


def request_llm2():
    template = f"请对以下内容意图识别，意图为[其他，重复警情, 警情画像, 电话分析]，直接输出意图名称，结构为数组。\n内容：xxx"
    json_data = {
        "content": template,
        "stream": True
    }

    url = "http://127.0.0.1:8080/llm/module_search"
    t1 = time.time()
    res = requests.post(url=url, json=json_data, headers=headers, stream=True)
    print(time.time() - t1)
    for line in res.iter_lines():
        print(line)

# request_llm()
request_llm2()
