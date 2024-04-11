# -*- coding:utf-8 -*-

"""
@date: 2024/3/27 下午3:33
@summary:
"""
from datasets import load_dataset

train_dataset = load_dataset(path="/media/mesie/a1d6502f-8a4a-4017-a9b5-3777bd223927/wikipedia/20220301.simple",
                 data_files="train-00000-of-00001.parquet")

train_dataset = train_dataset["train"]
print(train_dataset[0])