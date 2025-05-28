# -*- coding:utf-8 -*-

"""
@date: 2024/3/26 下午7:52
@summary:
"""
from tokenizer.bert_tokenizer import BertTokenizer

def sample_BertTokenizer():
    text = "“五一”小长假临近，30岁的武汉市民万昕在文旅博览会上获得了一些制定5天旅游计划的新思路。“‘壮美广西’‘安逸四川’，还有‘有一种叫云南的生活’这些展馆标识都很新颖，令人心向往之。”万昕说，感到身边越来越多的人走出家门去旅游。"
    # text = 'Say that thou didst forsake me for some fault, And I will comment upon that offence; Speak of my lameness, and I straight will halt, Against thy reasons making no defence.'

    tokenizer = BertTokenizer(vocab_file='../lib/albert_chinese_base/vocab.txt')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    plus = tokenizer.encode_plus(text, text_pair=text, max_len=100, padding=True)
    print("=" * 50, "自定义")
    print(tokens)
    print(ids)
    print(plus)


    official_tokenizer = BertTokenizer(vocab_file='../lib/albert_chinese_base/vocab.txt')
    o_tokens = official_tokenizer.tokenize(text)
    o_ids = official_tokenizer.convert_tokens_to_ids(o_tokens)
    o_plus = official_tokenizer.encode_plus(text, text_pair=text, max_len=100, padding='max_length', truncation='longest_first')
    print('=' * 50 + 'huggingface')
    print(o_tokens)
    print(o_ids)
    print(o_plus)

    assert tokens == o_tokens
    assert ids == o_ids
    assert plus['input_ids'] == o_plus['input_ids']
    assert plus['token_type_ids'] == o_plus['token_type_ids']
    assert plus['attention_mask'] == o_plus['attention_mask']

    print('=' * 50 + 'special tokens')
    tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    print(tokenizer.convert_tokens_to_ids(tokens))
    print(official_tokenizer.convert_tokens_to_ids(tokens))


if __name__ == "__main__":
    sample_BertTokenizer()