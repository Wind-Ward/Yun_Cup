# coding: utf8

from pathlib import Path
import re
import jieba

fill_value = "CSFxe"


def get_stop_words():
    path = Path(__file__).absolute().parent / 'stopwords.txt'
    with open(path) as f:
        words = [line.strip() for line in f]
        words.append(' ')
    return frozenset(words)


def get_yan_words():
    path = Path(__file__).absolute().parent / 'yan_word.txt'
    with open(path) as f:
        words = [line.strip() for line in f]
    return frozenset(words)

_yan_words = get_yan_words()
# def _get_yan_regex():
#     yan = r''
#     for y in get_yan_words():
#         g = y.replace('|', '\|')
#         yan += '|' + g
#     return yan[1:]
# _yan_regex = _get_yan_regex()


def clean_str(stri):
    stri = stri.lower()
    stri = re.sub(r'<br />n', '。', stri)   # 使用句号代替这个字符
    stri = re.sub(r'<br />', '。', stri)    # 句号代替这个字符
    stri = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
                  'CSFurl', stri)   # 替换掉url
    stri = re.sub(r'1\d{10,}', 'CSFphone', stri)  # 替换11位电话号码
    stri = re.sub(r'\d{1,10}', 'CSFnum', stri)    # 替换其余的数字
    stri = re.sub(r'\d{12,}', 'CSFnum', stri)
    # stri = re.sub(_yan_regex, 'CSFregex', stri)
    for w in _yan_words:
        stri = stri.replace(w, 'CSFyan')
    if stri == '':
        return fill_value
    return stri.strip()


def process_str(stri):
    word_list = jieba.cut(clean_str(stri))
    _filter_words = [w for w in word_list if w not in get_stop_words() and len(w) > 0]
    x = " ".join(_filter_words)
    return x
