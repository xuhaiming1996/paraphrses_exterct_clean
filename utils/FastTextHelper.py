#!/usr/bin/env python
import jieba
from fastText import load_model
from fastText import util
from scipy import spatial
import numpy as np



class FastTextHelper:
    def __init__(self,modelpath="../source/cc.zh.300.bin"):
        self.model = load_model(modelpath)
        self.word2vec_dim = 300

    def getWordVec(self, word):
        return self.model.get_word_vector(word)

    def _get_sen_vec(self, wordlist):
        vecs_pre = []
        for word in wordlist:
            v = self.model.get_word_vector(word)
            vecs_pre.append(v)

        vecs = np.array(vecs_pre)
        sen_vec = np.mean(vecs,axis=0)
        return sen_vec

    def get_cos_score(self,sen1, sen2):
        '''
        :param sen1: 这里是没有分好词
        :param sen2: 这里是没有分好词
        :return:
        '''
        words_sen1 = list(jieba.cut(sen1))
        words_sen2 = list(jieba.cut(sen2))
        sen_vec_1=self._get_sen_vec(words_sen1)
        sen_vec_2=self._get_sen_vec(words_sen2)
        sim = 1 - spatial.distance.cosine(sen_vec_1, sen_vec_2)
        return sim


# if __name__=="__main__":
#     fastTextHelper=FastTextHelper()
#     print(fastTextHelper.get_cos_score("我爱你","我爱你"))









