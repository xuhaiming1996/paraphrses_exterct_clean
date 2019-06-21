import csv
import numpy as np
from gensim.models import FastText
import chinese_seg_utils.segment_chinese as segment_ch
import pandas as pd
from scipy import spatial
import pickle

word2vec_dim = 300
a = 1e-3
fasttext_model = FastText.load('./fasttext_model/fasttextmodel_baiduzhidao_300_big_hmmseg')  # 词向量模型

with open("baiduzhidao_uut_hmm_a-3.pickle", "rb") as f:
    # u = pickle._Unpickler(f)
    # u.encoding = 'latin1'
    # p = u.load()
    uut = pickle.load(f,encoding = 'latin1')

with open("./word_count_baiduzhidao_hmm.pickle", "rb") as f:
    w_c = pickle.load(f)
total_words = 0
for k, v in w_c.items():
    total_words += v


def get_word_frequency(word_text):
    if word_text in w_c:
        return float(w_c[word_text]) / float(total_words)
    return 1e-10






def get_word_avg_vec(question):
    wordlist = list(segment_ch.extract_seg_word_hmm(question))
    question_vec = np.asarray([0.0] * word2vec_dim)
    try:
        for word in wordlist:
            a_value = a / (a + get_word_frequency(word))
            question_vec = np.add(question_vec, np.multiply(a_value, fasttext_model[word]))
        question_vec = question_vec / float(len(wordlist) + 0.000001)
        question_vec = np.subtract(question_vec, np.multiply(uut, question_vec))
        return question_vec
    except Exception:
        return np.asarray([0.0] * word2vec_dim)



def spatial_cos(v1, v2):
    sim = 1 - spatial.distance.cosine(v1, v2)
    return sim


def get_cos_score(sentence_a, sentence_b):
    v_a=get_word_avg_vec(sentence_a)
    v_b = get_word_avg_vec(sentence_b)
    return spatial_cos(v_a, v_b)


def find_paraphrase(file_a_name,file_b_name,file_save_name,limit_score):
    f_a=open(file_a_name,mode="r",encoding="utf-8")
    f_b=open(file_b_name,mode="r",encoding="utf-8")
    f_w=open(file_save_name,mode="w",encoding="utf-8")
    num=0
    for sentence_a,sentence_b in zip(f_a,f_b):
        sentence_a=sentence_a.replace(" ", "").replace("\n", "").replace("\r\n", "").replace("\t", "").replace("　　", "")
        sentence_b=sentence_b.replace(" ", "").replace("\n", "").replace("\r\n", "").replace("\t", "").replace("　　", "")
        f = get_cos_score(sentence_a, sentence_b)

        if f > limit_score:
            num+=1
            f_w.write("{}\t{}".format(sentence_a, sentence_b)+"\n")
            if num % 100 == 0:
                print("{}\t\t{}".format(sentence_a, sentence_b),"###","正在提取中....",num)

        else:
            pass
    f_a.close()
    f_b.close()
    f_w.close()






if __name__ == '__main__':
    file_a_name = "question1.txt"
    file_b_name = "question2.txt"
    file_save_name = "paraphrase_pairByVec.txt"
    find_paraphrase(file_a_name, file_b_name, file_save_name, 0.8)


