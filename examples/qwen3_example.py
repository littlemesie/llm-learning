# -*- coding:utf-8 -*-

"""
@date: 2025/5/28 下午4:10
@summary: Qwen3 0.6b 测试
"""
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/media/mesie/a1d6502f-8a4a-4017-a9b5-3777bd223927/model/qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
# prepare the model input
t = ""
label_list = ["暴力极端", "扬言极端", "上访", "家暴"]
prompt = f"你是一个文本分类专家，请对以下文进行分类，判读属于哪一个类别,并返回一个数组, 类别为：{label_list}, \n文本：{t}"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
t1 = time.time()
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
print(time.time() - t1)
