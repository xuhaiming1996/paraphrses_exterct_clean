'''

'''

import jieba
class ProproHelper:
    def __init__(self,filepath = "../src_tgt.txt"):
        self.filepath = filepath


    @staticmethod
    def _extractVocabsAndSentences(vocabfile="../miniparapair/vocab.txt",
                                   sentencesfile="../miniparapair/sentenceSet.txt"):
        '''

        :param vocabfile: 保存单词
        :param sentencesfile: 保存所有的句子
        :return:
        '''

