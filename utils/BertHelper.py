from bert_serving.client import BertClient
from tqdm import tqdm
import numpy as np
import  sys

class BertHelper:
    def __init__(self,port=8000):
       self.bc = BertClient(port=port)

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

    def get_cos_scores(self, sens_1, sens_2):
        '''

        :param sen1: 不需要分词,这里是多个句子
        :param sen2: 不需要分词 这里是多个句子
        :return:
        '''
        sen_vectors_1 = self.bc.encode(sens_1)
        sen_vectors_2 = self.bc.encode(sens_2)
        scores = self._cosine(sen_vectors_1,sen_vectors_2)
        return scores


    def isSatisfyBERT(self,sens_1, sens_2, threshold=0.900):
        '''

        :param sens_1: 这里的一个list的句子
        :param sens_2: 这里是一个list的句子
        :param threshold: 这里是阈值
        :return: 返回的也是一个list boolean
        '''
        scores = self.get_cos_scores(sens_1, sens_2)
        res=scores > threshold
        res=res.tolist()
        return res



class DataHelper:
    def __init__(self,filePath="../src_tgt_process_1.txt",batch_size =2048,total = 54087179):
        '''

        :param filePath:
        :param batch_size:
        :param total: 这个是用于进度条的
        '''
        self.filePath=filePath
        self.batch_size=batch_size
        self.total=total

    def batch_data(self):
        fr = open(self.filePath, mode="r", encoding="utf-8")
        pbar = tqdm(total=self.total, desc="wait,patient:")
        sens_1 = []
        sens_2 = []
        try:
            for line in fr:
                line = line.strip()
                if line != "":
                    tmp = line.split("---xhm---")
                    if tmp is None or tmp==[] or len(tmp)!=2 or tmp[0]=="" or tmp[1]=="":
                        print("出现错误数据，",line)
                        continue
                    else:
                        if len(sens_1)==self.batch_size:
                            yield sens_1,sens_2
                            pbar.update(self.batch_size)
                            sens_1,sens_2=[],[]
                        sens_1.append(tmp[0].strip())
                        sens_2.append(tmp[1].strip())
        except Exception:
            print("在这里出现错了")


        if len(sens_1) > 0:
            print("最后一个bacth_zie不够，大小为：",len(sens_1))
            yield sens_1,sens_2
            pbar.update(len(sens_1))
            return
        return



if __name__=="__main__":
    bertHelper = BertHelper(port=8100)
    dataHelper = DataHelper()
    batch_data = dataHelper.batch_data()

    fw = open("../src_tgt_process_bert_2.txt", mode="w", encoding="utf-8")


    while True:
        try:
            sens_1,sens_2=next(batch_data)
            if sens_1 == [] or sens_2 == []:
                print("出现为空的退出来.....")
                break
            else:
                flags = bertHelper.isSatisfyBERT(sens_1, sens_2, threshold=0.9)
                for sen_1,sen_2,flag in zip(sens_1,sens_2,flags):
                    if flag:
                        fw.write("---xhm---".join([sen_1,sen_2]) + "\n")

        except Exception:
            print("结束")
            break


    fw.close()





    # while True:
    #     sen1=input("请输入句子1：")
    #     sen2=input("请输入句子2：")

    '''
   下面测试 DataHelper 这个类
   '''





