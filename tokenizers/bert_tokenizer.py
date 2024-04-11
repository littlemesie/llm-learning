# -*- coding:utf-8 -*-

"""
@date: 2024/3/25 下午7:13
@summary:
"""
import torch
import unicodedata
from collections import OrderedDict
from tokenizers.bpe_tokenizer import WordPieceTokenizer

class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (bool):
            Whether to lowercase the input when tokenizing.
            Defaults to `True`.
        never_split (Iterable):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (bool):
            Whether to tokenize Chinese characters.
        strip_accents: (bool):
            Whether to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """
    def __init__(self, do_lower_case=False, never_split=None, tokenize_chinese_chars=True, strip_accents=True):
        """Constructs a BasicTokenizer."""
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split) if never_split else set()
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Tokenizes a piece of text using basic tokenizer.

        Args:
            text (str): A piece of text.
            never_split (List[str]): List of token not to split.

        Returns:
            list(str): A list of tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import BasicTokenizer
                basictokenizer = BasicTokenizer()
                tokens = basictokenizer.tokenize('He was a puppeteer')
                '''
                ['he', 'was', 'a', 'puppeteer']
                '''
        """
        text = self.convert_to_unicode(text)
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        orig_tokens = self.whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                if self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        output_tokens = " ".join(split_tokens).strip().split()
        return output_tokens

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a peice of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_split_on_punc(self, text, never_split):
        """Splits punctuation on a piece of text."""
        if text in never_split:
            return [text]
        output = [[]]
        for char in list(text):
            if self._is_punctuation(char):
                output.append([char])
                output.append([])
            else:
                output[-1].append(char)
        return ["".join(x) for x in output]

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        # We treat all non-letter/number ASCII as punctuation.
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if (cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or \
                (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or \
                (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or \
                (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F):
            return True
        return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            output.append(" {} ".format(char) if self._is_chinese_char(cp) else char)
        return "".join(output)

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            output.append(" " if self._is_whitespace(char) else char)
        return "".join(output)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        if char == "\t" or char == "\n" or char == "\r":  # treat them as whitespace
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        # \t, \n, and \r are technically control characters but we treat them as whitespace
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

class WordpieceTokenizer(object):
    """
    Runs WordPiece tokenization.

    Args:
        vocab (Vocab|dict):
            Vocab of the word piece tokenizer.
        unk_token (str):
            A specific token to replace all unknown tokens.
        max_input_chars_per_word (int):
            If a word's length is more than
            max_input_chars_per_word, it will be dealt as unknown word.
            Defaults to 100.
    """

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer`.

        Returns:
            list (str): A list of wordpiece tokens.

        Examples:
            .. code-block::

                from paddlenlp.transformers import BertTokenizer, WordpieceTokenizer

                berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                vocab  = berttokenizer.vocab
                unk_token = berttokenizer.unk_token

                wordpiecetokenizer = WordpieceTokenizer(vocab,unk_token)
                inputs = wordpiecetokenizer.tokenize("unaffable")
                print(inputs)
                '''
                ["un", "##aff", "##able"]
                '''
        """

        output_tokens = []
        for token in self.whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

    def whitespace_tokenize(self, text):
        """Runs basic whitespace cleaning and splitting on a peice of text."""
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

class BertTokenizer():
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, tokenizer_chinese_chars=True):
        self.special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.unk, self.sep, self.pad, self.cls, self.mask = self.special_tokens
        self.do_basic_tokenize = do_basic_tokenize
        self.vocab = self._load_vocab(vocab_file)
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case, self.special_tokens, tokenizer_chinese_chars)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab_size=len(self.vocab), lowercase=do_lower_case,
                                                      basic_tokenizer=lambda x: x.strip().split(),
                                                      unk=self.unk, sep=self.sep, pad=self.pad, cls=self.cls, mask=self.mask)
        self.wordpiece_tokenizer.load(vocab=self.vocab)

    def _load_vocab(self, vocab_file):
        vocab = OrderedDict()
        for idx, token in enumerate(open(vocab_file, 'r').readlines()):
            vocab[token.rstrip('\n')] = idx
        return vocab

    def tokenize(self, text):
        tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.special_tokens):
                if token in self.special_tokens:
                    tokens.append(token)
                else:
                    tokens.extend(self.wordpiece_tokenizer.tokenize(token, add_pre=None, add_mid="##", add_post=None))
        else:
            tokens = self.wordpiece_tokenizer.tokenize(text, add_pre=None, add_mid="##", add_post=None)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens, ]
        return [self.vocab.get(token, self.vocab.get(self.unk)) for token in tokens]

    def encode_plus(self, text, text_pair=None, max_len=1024, padding=True, truncation=True, truncation_side='right'):
        """
        返回input_ids, segment_ids, attention_mask
        padding: True, 将text、text_pair拼接后再pad到max_len长度
        trunction：True, 将text、text_pair拼接后如果超过max_len，按longest_first策略进行trunc
        """

        ############### tokenizer + tokens_to_ids ###############
        text_ids = self.convert_tokens_to_ids(self.tokenize(text))
        text_pair_ids = self.convert_tokens_to_ids(self.tokenize(text_pair)) if text_pair else []  # 在bert中支持输入2个text

        ############### trunction ###############
        ids_len = len(text_ids) + len(text_pair_ids) + 3 if text_pair_ids else len(text_ids) + 2
        if truncation and ids_len > max_len:
            for _ in range(ids_len - max_len):
                if len(text_ids) > len(text_pair_ids):  # TODO: 其他trunc策略
                    text_ids = text_ids[:-1] if truncation_side == 'right' else text_ids[1:]
                else:
                    text_pair_ids = text_pair_ids[:-1] if truncation_side == 'right' else text_pair_ids[1:]

        ############### 定制bert输入格式 ###############
        # 1个text，组织成[cls] text1 [sep]
        # 2个text，则组织成[cls] text1 [sep] text2 [sep]
        input_ids = self.convert_tokens_to_ids([self.cls]) + text_ids + self.convert_tokens_to_ids([self.sep])
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        if text_pair_ids:
            input_ids += text_pair_ids + self.convert_tokens_to_ids([self.sep])
            token_type_ids += [1] * (len(text_pair_ids) + 1)
            attention_mask += [1] * (len(text_pair_ids) + 1)

        ############### padding ###############
        while padding and len(input_ids) < max_len:
            input_ids += self.convert_tokens_to_ids(self.pad)
            token_type_ids += [0]
            attention_mask += [0]
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}