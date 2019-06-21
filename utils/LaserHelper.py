'''
使用基于句子语义来进一步清洗复述对
'''

import sys
import os
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source')
sys.path.append(LASER + '/source/lib')

from embed import SentenceEncoder, EncodeLoad, EncodeFile
from text_processing import Token, BPEfastApply
import numpy as np

class PrepareCorpusformat:
    def __init__(self):
        self.senSetCahe=set()
        self.bpe_codes=os.path.join(LASER,"models/93langs.fcodes")


    def _splitCorpusToSrcAndTgt(self,inputfile,src_tmp="../tmp/src.txt",tgt_tmp="../tmp/tgt.txt"):
        '''
        将 sen1---xhm---sen2
        分开存储
        :param inputfile:
        :param src_tmp:
        :param tgt_tmp:
        :return:
        '''
        fr = open(inputfile,mode="r",encoding="utf-8")

        fw_src = open(src_tmp,mode="w",encoding="utf-8")
        fw_tgt = open(tgt_tmp,mode="w",encoding="utf-8")

        for line in fr:
            line = line.strip()
            if line != "":
                src,tgt=line.split("---xhm---")
                fw_src.write(src+"\n")
                fw_tgt.write(tgt+"\n")
        fr.close()
        fw_src.close()
        fw_tgt.close()

    def _token(self, src_input="../tmp/src.txt",
              tgt_input="../tmp/tgt.txt",
              src_tmp="../tmp/src.tok",
              tgt_tmp="../tmp/tgt.tok"):

        Token(src_input,
              src_tmp,
              lang="zh",
              romanize=False,
              lower_case=True,
              verbose=True)

        Token(tgt_input,
              tgt_tmp,
              lang="zh",
              romanize=False,
              lower_case=True,
              verbose=True)


    def _bpe(self,src_input="../tmp/src.tok",
                 tgt_input="../tmp/tgt.tok",
                 src_tmp  ="../tmp/src.bpe",
                 tgt_tmp  ="../tmp/tgt.bpe"):

        BPEfastApply(src_input,
                     src_tmp,
                     self.bpe_codes,
                     verbose=True, over_write=False)

        BPEfastApply(tgt_input,
                     tgt_tmp,
                     self.bpe_codes,
                     verbose=True, over_write=False)

    def _mergeSrcAndTgtToCorpus(self,
                               src_tmp="../tmp/src.bpe",
                               tgt_tmp="../tmp/tgt.bpe",
                               outputfile="../src_tgt_bpe_3.txt"):


        fr_src = open(src_tmp,mode="r",encoding="utf-8")
        fr_tgt = open(tgt_tmp,mode="r",encoding="utf-8")


        fw =open(outputfile,mode="w",encoding="utf-8")

        for src,tgt in zip(fr_src,fr_tgt):
            src=src.strip()
            tgt=tgt.strip()
            assert src!=""
            assert tgt!=""
            newline="---xhm---".join([src,tgt])
            fw.write(newline+"\n")

        fw.close()
        fr_src.close()
        fr_tgt.close()


    def corpusFormat(self,inputfile):
        # 第一步 将原始语料分开
        self._splitCorpusToSrcAndTgt(inputfile)
        print("***************文件分割成功***************")
        self._token()
        print("***************文件分词成功***************")
        self._bpe()
        print("***************文件bpe成功***************")
        self._mergeSrcAndTgtToCorpus()
        print("***************文件合并成功*************")

    '''
    def corpusMergeFormat(self, outfile):
        fr = open(self.inputfile, mode="r", encoding="utf-8")
        fw = open(outfile, mode="w", encoding="utf-8")
        num = 0
        for line in fr:
            num += 1
            if num % 10000 == 0:
                print("************数据正在处理中************", num / 28058691)
            try:
                line = line.strip()
                if line is not None and line != "":
                    sens = line.split("---xhm---")
                    assert len(sens) == 2
                    if sens[0] not in self.senSetCahe and sens[1] not in self.senSetCahe:
                        if len(self.senSetCahe) != 0:
                            # 将这一批的数据写入文件
                            newline = "---xhm---".join(self.senSetCahe) + "\n"
                            fw.write(newline)
    
                            Flag = "n"
                            while Flag != "y":
                                Flag = input("继续请输入y:")
                                print(self.senSetCahe)
    
                            self.senSetCahe.clear()
    
                    self.senSetCahe.add(sens[0])
                    self.senSetCahe.add(sens[1])
            except Exception:
                print("处理出现错误，将错误的全部丢弃掉")
                self.senSetCahe.clear()
    '''




class LaserHelper:
    def __init__(self):
        self.modelpath = os.path.join(LASER, "models/bilstm.93langs.2018-12-26.pt")
        # 不需要设置batch_size 根据max_tokens自动调整
        self.encoder = SentenceEncoder(self.modelpath,
                                  max_sentences=None,
                                  max_tokens=12000,
                                  sort_kind='quicksort',
                                  cpu=False)
        self.batch_size = 64

    def _cosine(self,vectors_1, vectors_2):
        '''
        :param vector1s:是一个二维的numpy数组
        :param vector2s:是一个二维的numpy数组
        :return: 返回的是一个list
        '''
        molecule = np.sum(vectors_1 * vectors_2, axis=1)
        denominator_1 = np.linalg.norm(vectors_1, axis=1)
        denominator_2 = np.linalg.norm(vectors_2, axis=1)
        denominator = denominator_1 * denominator_2
        scores = molecule / denominator
        return scores

    def _getSensvec(self,sens):
        '''
        :param sens: 是一个句子的列表
        :return:  返回的是每个句子的的句子向量
        '''
        res = self.encoder.encode_sentences(sens)
        return res


    def _calCosine(self,sens_1,sens_2):
        senvecs_1=self._getSensvec(sens_1)
        senvecs_2=self._getSensvec(sens_2)
        scores=self._cosine(senvecs_1,senvecs_2)

        return scores

    def _calL2(self, sens_1, sens_2):
        senvecs_1 = self._getSensvec(sens_1)
        senvecs_2 = self._getSensvec(sens_2)
        scores = np.sqrt(np.sum(np.square(senvecs_1 - senvecs_2),axis=1))
        return scores


    def calL2andSave(self,src_tmp="../tmp/src.bpe",
                                   tgt_tmp="../tmp/tgt.bpe",
                                   outputfile="../src_tgt_scores_laser.txt"):
        fr_src = open(src_tmp,mode="r",encoding="utf-8")
        fr_tgt = open(tgt_tmp,mode="r",encoding="utf-8")

        fw = open(outputfile,mode="w",encoding="utf-8")

        sens_1 = []
        sens_2 = []

        for src,tgt in zip(fr_src,fr_tgt):
            src=src.strip()
            tgt=tgt.strip()
            assert src!=""
            assert tgt!=""
            if len(sens_1) == self.batch_size:
                scores = self._calL2(sens_1,sens_2)
                for s1,s2,s in zip(sens_1,sens_2,scores):
                    s1=s1.replace("@@","").replace(" ","")
                    s2=s2.replace("@@","").replace(" ","")
                    newline="---xhm---".join([s1,s2,str(s)])+"\n"
                    fw.write(newline)

                sens_1 = []
                sens_2 = []

            sens_1.append(src)
            sens_2.append(tgt)


        if len(sens_1) > 0:
            scores = self._calL2(sens_1, sens_2)
            for s1, s2, s in zip(sens_1, sens_2, scores):
                s1 = s1.replace("@@", "").replace(" ", "")
                s2 = s2.replace("@@", "").replace(" ", "")
                newline = "---xhm---".join([s1, s2, str(s)]) + "\n"
                fw.write(newline)
        fr_tgt.close()
        fr_src.close()
        fw.close()

    def extractParaphrasePair(self,
                              inputfile = "../src_tgt_scores_laser.txt",
                              outputfile = "../src_tgt_",
                              thresold=0.3):

        outputfile = outputfile+str(thresold)+".txt"

        fw = open(outputfile, mode="w", encoding="utf-8")

        with open(inputfile,mode="r",encoding="utf-8") as fr:
            for line in fr:
                line=line.strip()
                if line != "":
                    sen1,sen2,score=line.split("---xhm---")
                    if float(score)<thresold:
                        fw.write("---xhm--".join([sen1,sen2])+"\n")

        fw.close()



if __name__=="__main__":
    laserHelper = LaserHelper()
    prepareCorpusformat = PrepareCorpusformat()
    prepareCorpusformat.corpusFormat("../src_tgt_process_bert_2.txt")

    laserHelper.calL2andSave()
    laserHelper.extractParaphrasePair()


