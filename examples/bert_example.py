# -*- coding:utf-8 -*-

"""
@date: 2024/4/15 下午7:01
@summary: bert 模型训练
"""
import torch
import logging
from core.system_config import project_dir
from core.log import configure_logging
from transformers.bert.config import BertConfig
from transformers.bert.task_training import BertForMaskedLM

class ModelConfig(object):
    def __init__(self):
        self.pretrained_model_dir = f"{project_dir}/lib/albert_chinese_base"
        self.use_embedding_weight = True  # 是否使用Token embedding中的权重作为预测时输出层的权重
        # 把原始bert中的配置参数也导入进来
        configure_logging(path='bert')
        bert_config = BertConfig.from_json_file(f"{project_dir}/lib/albert_chinese_base/config.json")
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")

def make_data():
    import numpy as np
    ids = np.random.random_integers(0, 20000, 512 * 3).reshape(3, 512)
    input_ids = torch.tensor(ids).transpose(0, 1)
    labels = np.random.random_integers(0, 1, 512 * 3).reshape(3, 512)
    masked_lm_labels = torch.tensor(labels, dtype=torch.long).transpose(0, 1)  # [src_len,batch_size]
    return input_ids, masked_lm_labels


if __name__ == '__main__':
    config = ModelConfig()
    input_ids, masked_lm_labels = make_data()

    model = BertForMaskedLM(config)
    output = model(input_ids=input_ids,
                   masked_lm_labels=None)
    print(output)
    output = model(input_ids=input_ids,
                   masked_lm_labels=masked_lm_labels)
    print(output)
