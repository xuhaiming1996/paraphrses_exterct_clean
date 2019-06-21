import jieba

from pybloom_live import BloomFilter
from FastTextHelper import FastTextHelper
import torch
import torch.nn as nn
import numpy as np

import os
print("\n加载Faxtext model")
fastTextHelper = FastTextHelper()
print("加载Faxtext model完毕\n")

import faiss
import datetime



'''

我根据第一步句子处理的结果发现句子有 2千万条
根据我们这条件我觉得可以加载到内存
最暴力的将所有的 句子向量全部加载到内存

'''

class MiniParaHelperFastText:

    def __init__(self, wordVecfile   = "../miniparapair/fastText/wordVec.txt",
                       sentencesfile ="../miniparapair/fastText/sentenceSet.txt",
                       k   = 5,
                       dim = 300,
                       sens_scores_jaccard="../miniparapair/fastText/sens_scores_jaccard.txt",
                       save_index_file="../miniparapair/fastText/index_faiss"):


        # self.vocabfile     = vocabfile
        self.wordVecfile       = wordVecfile
        self.sentencesfile = sentencesfile
        self._createEmbedding()
        self.word2id = {word:id for id ,word in enumerate(self.vocab)}
        self._createSenSetVecsNumpy()  # 将文件中的所有句子全部加到内存 同时生成每个句子的句子向量
        self.k = k
        self.dim = dim
        self.sens_scores_jaccard = sens_scores_jaccard
        self.save_index_file=save_index_file
        self._initOrLoad_faissIndex()

    def _createEmbedding(self):

        start_time = datetime.datetime.now()  # 放在程序开始处
        vocab=[]
        wordvecs=[]
        print("\n开始提取单词向量....")
        with open(self.wordVecfile,mode="r",encoding="utf-8") as fr:
            for line in fr:
                line=line.strip()
                if line!="":
                    tokens = line.split(" ")
                    vocab.append(tokens[0])
                    vec = tokens[1:]
                    vec = [float(num) for num in vec]
                    assert len(vec)==300
                    wordvecs.append(vec)
        print("结束提取单词向量....\n")
        print("\n开始创建Embedding....")
        weight = torch.FloatTensor(wordvecs)
        self.embedding = nn.Embedding.from_pretrained(weight)
        print("结束创建Embedding....\n")

        self.vocab=vocab
        print("共有单词：",len(self.vocab))
        end_time = datetime.datetime.now()  # 放在程序结尾处
        interval = (end_time - start_time).seconds  # 以秒的形式
        print("提取+词嵌入创建完毕，共用时：",interval)


    def _convertWords2ids(self, wordlist):
        '''
        将单词转换为对应的id
        :param wordlist:
        :return:
        '''
        ids = []
        for word in wordlist:
            if word not in self.word2id:
                print("注意:有单词不存在单词表里",word,"这是。。。无奈")
                continue
            ids.append(self.word2id.get(word))

        return ids


    def _createSenSetVecsNumpy(self):

        '''
        将所有句子的语料全部加载到内存
        :return:
        '''
        print("\n开始生成所有句子的句子向量....")
        start_time = datetime.datetime.now()  # 放在程序开始处
        self.sens = []    # 用来所有的存储句子      我 爱 你
        senVecs   = []    # 用来存储所有句子向量
        with open(self.sentencesfile,mode="r",encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if line != "":
                    tokens = line.split(" ")
                    tokens_ids = self._convertWords2ids(tokens)
                    self.sens.append(line)
                    senVecs.append(self._getSenVec(tokens_ids))


        self.senVecs = np.ascontiguousarray(senVecs) # 是一个numpy

        print("\n开始执行_normalize_L2")
        faiss.normalize_L2(self.senVecs)
        print("\n结束执行_normalize_L2")


        print("\n结束生成所有句子的句子向量....")

        end_time = datetime.datetime.now()  # 放在程序结尾处
        interval = (end_time - start_time).seconds  # 以秒的形式
        print("句子向量生成完毕，共用时：",interval)


    def _getSenVec(self,wordids):
        '''

        :param wordids: 一句话的单词的ids
        :return: 返回这句话的句子向量
        '''
        wordids = torch.LongTensor(wordids)
        wordvecs = self.embedding(wordids).data.numpy()
        sen_vec = np.mean(wordvecs, axis=0)
        return sen_vec

    def _indexLoad(self, nprobe=300, gpu=False):
        print('Reading FAISS index')
        print(' - index: {:s}'.format(self.save_index_file))
        index = faiss.read_index(self.save_index_file)
        print(' - found {:d} sentences of dim {:d}'.format(index.ntotal, index.d))
        print(' - setting nbprobe to {:d}'.format(nprobe))
        if gpu:
            print(' - transfer index to %d GPUs ' % faiss.get_num_gpus())
            # co = faiss.GpuMultipleClonerOptions()
            # co.shard = True
            index = faiss.index_cpu_to_all_gpus(index)  # co=co
            faiss.GpuParameterSpace().set_index_parameter(index, 'nprobe', nprobe)
        self.index = index


    def _initOrLoad_faissIndex(self):
        if os.path.exists(self.save_index_file):
            # 已经存在 我们直接load
            self._indexLoad()
        else:
            index = faiss.IndexFlatL2(self.dim)
            # faiss.normalize_L2(self.senVecs)
            # index.add(np.ascontiguousarray(self.senVecs))
            index.add(self.senVecs)
            print(' - saving index into ' + self.save_index_file)
            faiss.write_index(index, self.save_index_file)
            self.index=index


    def _jaccard_Similarity(self,sen1,sen2):
        '''
        两个集合的交集长度除以两个集合的并集的长度
        :param sen1:
        :param sen2:
        :return:
        '''
        sen1 =set(sen1)
        sen2 =set(sen2)
        return len(sen1 & sen2)/len(sen1 | sen2)



    def buffered_read(self, batch_size):
        '''
        batch 处理
        :param batch_size:
        :return:
        '''
        buffer_sens = []
        buffer_sensvecs = []

        for sen, senvec in zip(self.sens, self.senVecs):
            buffer_sens.append(sen)
            buffer_sensvecs.append(senvec)
            if len(buffer_sens) >= batch_size:
                yield buffer_sens,buffer_sensvecs
                buffer_sens = []
                buffer_sensvecs = []

        if len(buffer_sens) > 0:
            yield buffer_sens, buffer_sensvecs

    def _writer(self,fw,sens,D,I):
        for query_sen,res_i,d_i in zip(sens,I,D):
            fw.write("\n" + query_sen + "\n")
            for d, i in zip(d_i, res_i):
                newline = [self.sens[i], str(d), str(self._jaccard_Similarity(query_sen.split(" "), self.sens[i].split(" ")))]
                fw.write("---xhm---".join(newline) + "\n")


    def mini_paraPhrasePairAndSave(self,batch_size=2000,
                                       threshold_L2=0.3,
                                       threshold_jaccard=0.3):
        fw = open(self.sens_scores_jaccard,mode="w",encoding="utf-8")
        # flag = "y"
        num=0
        print("\n\n开始minParapair数据......")


        for sens, sensvecs in self.buffered_read(batch_size=batch_size):


            start_time = datetime.datetime.now()  # 放在程序开始处

            # start_time = datetime.datetime.now()  # 放在程序开始处
            sensvecs = np.ascontiguousarray(sensvecs)
            # print("\n 一个batch_size的数据准备完毕，用时：",(datetime.datetime.now() - start_time).seconds)

            # start_time = datetime.datetime.now()  # 放在程序开始处
            D, I = self.index.search(sensvecs, self.k)
            # print("\n  一个batch_size的数据查询完毕，用时：", (datetime.datetime.now() - start_time).seconds)

            D = D.tolist()  # 这是最近的5个距离
            I = I.tolist()  # 这是最近的5个下标

            # start_time = datetime.datetime.now()  # 放在程序开始处
            self._writer(fw,sens,D,I)
            # print("\n  一个batch_size的数据写入完毕，用时：", (datetime.datetime.now() - start_time).seconds)

            # print("\n  一个batch_size的数据处理完毕，用时：", (datetime.datetime.now() - start_all).seconds)

            num += batch_size
            if num%10000==0:
                print("数据正在处理中...",
                          "*****",
                          "batch_size:", str(batch_size),
                          "*****",
                           "已处理： ",
                           num / 27054290,
                          "*****",
                           "一个batch_size的数据处理完毕，用时(单位 sec)：",
                          (datetime.datetime.now() - start_time).seconds)






            flag = "N"


class ProproHelper:

    def __init__(self,filepath      = "../src_tgt.txt",
                      vocabfile     = "../miniparapair/fastText/vocab.txt",
                      wordVecfile   = "../miniparapair/fastText/wordVec.txt",
                      sentencesfile ="../miniparapair/fastText/sentenceSet.txt",
                      maxlen = 25,
                      capacity=250000000):

        self.filepath      = filepath
        self.vocabfile     = vocabfile
        self.wordVecfile       = wordVecfile
        self.sentencesfile = sentencesfile
        self.maxlen        = maxlen
        self.bf            = BloomFilter(capacity=capacity)


    def extractVocabsAndSentences(self):
        '''
        这里考虑到词典需要100%准确，所以词典采用集合的方式去重
        这里句子 采用布隆过滤器 进行去重 随时一点精度
        :param vocabfile: 保存单词
        :param sentencesfile: 保存所有的句子
        :return:
        '''
        vocabSet = set()
        sentencesTokenSet=[]  #这是存储所有已经分好词句子 里面没有重复的
        num = 0

        try:
            with open(self.filepath, mode="r", encoding="utf-8") as fr:
                    for line in fr:
                        try:
                            num+=1
                            if num%100000==0:
                                print("数据正在提取单词，数据正在去重，，，",num/233864191)
                            line = line.strip()
                            if line != "":
                                sen1, sen2 = line.split("---xhm---")
                                if len(sen1) > self.maxlen or len(sen2) > self.maxlen:
                                    # 长度太大的不需要
                                    continue
                                words_1 = list(jieba.cut(sen1))
                                words_2 = list(jieba.cut(sen2))

                                # 将单词添加到单词集合中
                                for word in words_1:
                                    if word not in vocabSet:
                                        vocabSet.add(word)
                                for word in words_2:
                                    if word not in vocabSet:
                                        vocabSet.add(word)

                                #将句子添加到句子集合中
                                if sen1 not in self.bf:
                                    sentencesTokenSet.append(" ".join(words_1))
                                    self.bf.add(sen1)

                                if sen2 not in self.bf:
                                    sentencesTokenSet.append(" ".join(words_2))
                                    self.bf.add(sen2)
                        except Exception:
                            print("这是出错的行",line)
        except Exception:
               print("内部错误")




        with open(self.vocabfile,mode="w",encoding="utf-8") as fw:
            fw.write("\n".join(vocabSet))

        with open(self.sentencesfile,mode="w",encoding="utf-8") as fw:
            fw.write("\n".join(sentencesTokenSet))


    def computeAndSaveWord2vec(self):
        fr = open(self.vocabfile,mode="r",encoding="utf-8")
        fw = open(self.wordVecfile,mode="w",encoding="utf-8")
        for line in fr:
            line=line.strip()
            if line!="":
                vec = fastTextHelper.getWordVec(line)
                vec = [str(num) for num in vec]
                fw.write(line+" "+" ".join(vec)+"\n")

        fr.close()
        fw.close()


if __name__=="__main__":
    # proproHelper = ProproHelper()
    # proproHelper.extractVocabsAndSentences()
    # print("数据去重处理完毕")
    # print("为每一个单词计算词向量")
    # proproHelper.computeAndSaveWord2vec()
    # print(fastTextHelper.getWordVec("许海明"))

    '''
    以上代码注释掉 就不在执行
    '''
    miniParaHelperFastText=MiniParaHelperFastText()

    miniParaHelperFastText.mini_paraPhrasePairAndSave()










