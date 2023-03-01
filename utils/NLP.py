# coding=UTF-8
# NLP基础工具包

'''
@File: NLP
@Author: WeiWei
@Time: 2023/2/19
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import re
import jieba
import jieba.posseg as psg
import string

remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()


# 判断是否包含可分词的汉字
def language(text):
    # English
    # s = unicode(s)   # python2需要将字符串转换为unicode编码，python3不需要
    s = filter_str(text)
    result = []
    s = re.sub('[0-9]', '', s).strip()
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) > 0:
        result.append('en')
    if len(res2) <= 0:
        return 'en'

    # Chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, text)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u4e00-\u9fa5]+", '', text).strip()
    if len(res) > 0:
        result.append('cn')
    if len(res2) <= 0:
        return 'cn'

    # Korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, text)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\uac00-\ud7ff]+", '', text).strip()
    if len(res) > 0:
        result.append('ko')
    if len(res2) <= 0:
        return 'ko'

    # Jananese
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, text)  # 查询出所有的匹配字符串
    res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', text).strip()
    if len(res) > 0:
        result.append('ja')
    if len(res2) <= 0:
        return 'ja'
    return ','.join(result)


# 分词，目前用jieba分词工具
# # 全模式分词，把句子中所有可以成词的词语都扫描出来，词语会重复，且不能解决歧义，适合关键词提取
# words = jieba.cut(sent, cut_all=True)
# # 精确模式分词，将句子最精确的切分，此为默认模式，适合文本分析
# # 默认模式调用，可以忽略cut_all参数，写法如下：
# # seg_list = jieba.cut(sent)
# seg_list = jieba.cut(sent, cut_all=False)
# # 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
# seg_list = jieba.cut_for_search(sent)
def cut_org(text, cuttype=False):
    if text == "":
        return
    if cuttype:
        words = list(jieba.cut(text, cut_all=True))
    else:
        words = list(jieba.cut(text, cut_all=False))
    return words


# 分词，且过滤停用词
def cut_rm_stopwords(text, cuttype=False):
    words = filter_stopwords(cut_org(text, cuttype))
    return words


# 加载停用词词典
def load_stopwords(path='./dictionary/stop_words.txt'):
    stopwords = []
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in f:
            if line.strip() not in stopwords:
                stopwords.append(line.strip())
    return stopwords


# 过滤停用词
def filter_stopwords(words):
    stopwords = load_stopwords()
    if words[0].find('/') > 0:
        words = [x for x in words if x[:x.find('/')] not in stopwords]
    else:
        words = [x for x in words if x not in stopwords]
    return words


# 分词且带词性（pos：part of speech）
def cut_with_pos(text):
    words_pos = psg.cut(text)
    words_pos = ['{0}/{1}'.format(w, pos) for w, pos in words_pos]
    return words_pos


# 分词且带词性（pos：part of speech），并过滤停用词
def cut_with_pos_rm_sw(text):
    words_pos = psg.cut(text)
    words_pos = ['{0}/{1}'.format(w, pos) for w, pos in words_pos]
    stopwords = load_stopwords()
    words_pos = [x for x in words_pos if x[:x.find('/')] not in stopwords]
    return words_pos


# 分词且带词性（pos：part of speech），并过滤停用词，输出格式为：[['word', 'pos']]
def cut_with_pos_rm_sw_list(text):
    words_pos = psg.cut(text)
    stopwords = load_stopwords()
    words_pos = [[word, pos] for word, pos in words_pos if word not in stopwords]
    return words_pos


# word list过滤长度为1，只取标签为n开头的词，例如：名词、人名、地名、机构团体名、其它专有名词
def word_pos_filter(words):
    filter_words = [word for word, pos in words if pos.startswith('n') and len(word) > 1]
    return filter_words


# 计算text中词的词频，且默认会过滤停用词，因为绝大多数场景，停用词无实际意义
# sort定义是否需要排序输出
def calc_TF(text, sort=False):
    words = cut_rm_stopwords(text)
    words_tf = {}
    for w in words:
        words_tf[w] = words_tf.get(w, 0) + 1
    if sort:
        return sorted(words_tf.items(), key=lambda x: x[1], reverse=True)
    else:
        return words_tf


def load_dictionary(path='./dictionary/dictionary.txt'):
    # print("load dictionary start_time: {0}".format(datetime.datetime.now()))
    dic = []
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in f:
            dic.extend(word for word in line.strip().split(" "))
    print("loaded dictionary has {0} words.".format(len(dic)))
    # print("load dictionary end_time: {0}".format(datetime.datetime.now()))
    return dic


def intent_judge(text):
    intention_dic = []
    path = './dictionary/intention_dic.txt'
    with open(path, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in f:
            intention_dic.extend(word for word in line.strip())
    # print("loaded dictionary has {0} words.".format(len(intention_dic)))
    # 简单判断输入text是chat意图还是非chat意图
    for w in intention_dic:
        if w in text:
            return 'chat'
        else:
            continue
    return 'write'


def is_company_prompt(text):
    dic = './dictionary/company_dic.txt'
    company_prompt_dic = []
    with open(dic, 'r', encoding='UTF-8', errors='ignore') as f:
        for line in f:
            company_prompt_dic.extend(word for word in line.strip())
    for w in company_prompt_dic:
        if w in text:
            return True
        else:
            continue
    return False
