from typing import List, Union

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

from nltk import sent_tokenize, wordpunct_tokenize
from razdel import sentenize, tokenize
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def calculate_sentence_length(sentence: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]) -> int:
    return len(tokenizer.tokenize(sentence, add_special_tokens=True))


def split_sentence(long_sentence: str, max_seq_len: int, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                   lang: str = 'ru') -> List[str]:
    if calculate_sentence_length(long_sentence, tokenizer) <= max_seq_len:
        return [long_sentence]
    word_bounds = []
    if lang.lower() in {'en', 'eng', 'english'}:
        start_pos = 0
        for cur_word in wordpunct_tokenize(long_sentence):
            found_idx = long_sentence[start_pos:].find(cur_word)
            if found_idx < 0:
                raise ValueError(f'The token "{cur_word}" is not found in the text "{long_sentence}".')
            word_bounds.append((found_idx + start_pos, found_idx + start_pos + len(cur_word)))
            start_pos = found_idx + start_pos + len(cur_word)
    else:
        word_bounds = [(it.start, it.stop) for it in tokenize(long_sentence)]
    if len(word_bounds) < 2:
        return [long_sentence]
    middle_idx = (len(word_bounds) - 1) // 2
    first_sentence_start = word_bounds[0][0]
    first_sentence_end = word_bounds[middle_idx][1]
    second_sentence_start = word_bounds[middle_idx + 1][0]
    second_sentence_end = word_bounds[-1][1]
    sentences = split_sentence(long_sentence[first_sentence_start:first_sentence_end], max_seq_len, tokenizer, lang)
    sentences += split_sentence(long_sentence[second_sentence_start:second_sentence_end], max_seq_len, tokenizer, lang)
    return sentences


def split_text_into_sentences(source_text: str, newline_as_separator: bool = True, lang: str = 'ru',
                              max_seq_len: int = 512,
                              tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, None] = None) -> List[str]:
    if lang.strip().lower() not in {'ru', 'rus', 'russian', 'en', 'eng', 'english'}:
        raise ValueError(f'The language {lang} is not supported!')
    if newline_as_separator:
        paragraphs = list(map(
            lambda it3: ' '.join(it3.split()).strip(),
            filter(
                lambda it2: len(it2) > 0,
                map(
                    lambda it1: it1.strip(), source_text.split('\n')
                )
            )
        ))
    else:
        prepared_text = ' '.join(source_text.split()).strip()
        if len(prepared_text) == 0:
            paragraphs = []
        else:
            paragraphs = [prepared_text]
    if len(paragraphs) == 0:
        return []
    sentences = []
    if lang.strip().lower() in {'ru', 'rus', 'russian'}:
        for cur_paragraph in paragraphs:
            for it in sentenize(cur_paragraph):
                new_sentence = it.text.strip()
                if len(new_sentence) > 0:
                    sentences.append(new_sentence)
    else:
        for cur_paragraph in paragraphs:
            for it in sent_tokenize(cur_paragraph):
                new_sentence = it.strip()
                if len(new_sentence) > 0:
                    sentences.append(new_sentence)
    if tokenizer is None:
        return sentences
    sentences_ = []
    for cur_sentence in sentences:
        sentences_ += split_sentence(cur_sentence, max_seq_len, tokenizer, lang)
    return sentences_
