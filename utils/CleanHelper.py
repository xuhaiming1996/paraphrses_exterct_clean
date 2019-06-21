import math
from rouge import Rouge
# import FastTextHelper
# fastTextHelper = FastTextHelper(modelpath="../source/cc.zh.300.bin")

class CleanHelper:

    @staticmethod
    def isSatisfyNoEqual(sen1,sen2):
        '''
        :param sen1: 问题1
        :param sen2: 问题2
        :return: bool
        '''
        if sen1==sen2:
            return False
        return True

    @staticmethod
    def isSatisfyMaxLenAndMinLen(sen1,sen2,maxlen=25,minlen=6):
        '''
        只保留在[minlen,maxlen] 之间的语料
        :param sen1: 问题1
        :param sen2: 问题2
        :return: bool
        '''
        len1 = len(sen1)
        len2 = len(sen2)
        if len1 > maxlen or len2 > maxlen:
            return False
        if len1 < minlen or len2 < minlen:
            return False
        return True

    @staticmethod
    def isSatisfyLongthDiff(sen1,sen2,threshold=0.4):
        '''
        长度差是否满足阈值要求， 不满足返回False  满足返回True
        长度差不能超过min(len(sen1),len(sen2))中的多少百分比
        即：
          距离差太大的不要
        :param sen1:
        :param sen2:
        :param threshold:
        :return:
        '''
        len1 = len(sen1)
        len2 = len(sen2)
        if abs(len1-len2)/min(len2, len1) >= threshold:
            return False
        return True

    @staticmethod
    def isSatisfyLCS(sen1,sen2,threshold=0.65):
        '''
        判断两个句子之间的最长公共子串，若最短长度比例大于threshold则返回True
        即 公共子串太高的不要

        :param sen1:
        :param sen2:
        :param threshold:
        :return:
        '''
        mmax = CleanHelper._find_LCS(sen1, sen2)
        len1 = len(sen1)
        len2 = len(sen2)
        min_len = min(len1,len2)
        if mmax/min_len>threshold:
            return False
        return True

    @staticmethod
    def isSatisfyTER(sen1,sen2,threshold=5):
        '''
        判断两个句子之间的编辑距离，若编辑距离太小则不要
        :param sen1:
        :param sen2:
        :param threshold:
        :return:
        '''
        min_TER = CleanHelper._minEditDistance(sen1,sen2)
        if min_TER <= threshold:
            return False
        return True

    @staticmethod
    def isSatisfyROUGE(sen1,sen2,threshold):
        '''
        计算两个句子之间的rouge,低于threshold的全部不要
        rouge 太低的不要
        :param sen1:
        :param sen2:
        :param threshold:
        :return:
        '''
        score = CleanHelper._get_rouge_score(sen1,sen2)
        if score <= threshold:
            return False
        return True

    @staticmethod
    def _get_rouge_score(sentence_a, sentence_b):
        rouge = Rouge()
        weight = [0.33, 0.33, 0.33]  # 分别代表rouge-1， rouge-2， rouge-l所占比重。
        sentence_a = " ".join([word for word in sentence_a])
        sentence_b = " ".join([word for word in sentence_b])
        try:
            scores = rouge.get_scores(sentence_a, sentence_b, avg=True)
        except Exception:
            return 0
        rouge_1 = scores["rouge-1"]
        rouge_2 = scores["rouge-2"]
        rouge_l = scores["rouge-l"]
        f = rouge_1["f"] * weight[0] + rouge_2["f"] * weight[0] + rouge_l["f"] * weight[0]
        return f

    @staticmethod
    def _minEditDistance(word1, word2):
        """
        编辑距离
        :type word1: str
        :type word2: str
        :rtype: int
        """
        len_word1 = len(word1)
        len_word2 = len(word2)
        dp = []
        for row in range(len_word1 + 1):
            this_tmp = []
            for col in range(len_word2 + 1):
                if row == 0:
                    this_tmp.append(col)
                elif col == 0:
                    this_tmp.append(row)
                else:
                    this_tmp.append(False)
            dp.append(this_tmp)
        # print(dp)
        for row in range(1, len_word1 + 1):
            for col in range(1, len_word2 + 1):
                if word1[row - 1] == word2[col - 1]:
                    dp[row][col] = dp[row - 1][col - 1]
                else:
                    dp[row][col] = min(dp[row - 1][col], dp[row - 1][col - 1], dp[row][col - 1]) + 1
        return dp[len_word1][len_word2]

    @staticmethod
    def _find_LCS(s1, s2):
        '''
        求最长公共子串
        :param s1:
        :param s2:
        :return:
        '''
        m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
        mmax = 0  # 最长匹配的长度
        p = 0  # 最长匹配对应在s1中的最后一位
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    m[i + 1][j + 1] = m[i][j] + 1
                    if m[i + 1][j + 1] > mmax:
                        mmax = m[i + 1][j + 1]
                        p = i + 1
        return  mmax  # 最长匹配的长度




def isParaphrase_pair(sen1,sen2):
    if CleanHelper.isSatisfyNoEqual(sen1,sen2):
        if CleanHelper.isSatisfyMaxLenAndMinLen(sen1,sen2):
            if CleanHelper.isSatisfyLongthDiff(sen1,sen2,threshold=0.35):
                if CleanHelper.isSatisfyLCS(sen1,sen2,threshold=0.65):
                    if CleanHelper.isSatisfyROUGE(sen1,sen2,threshold=0.3):
                        return True
                    else:
                        return False #print("两个句子的rouge太低")
                else:
                    return False #print("两个句子的最长公共子串不符合")
            else:
                return False #print("两个句子的长度差，不符合")
        else:
            return False #print("两个句子的长度，不符合")
    else:
        return False #print("两个句子相等，不符合")





if __name__=="__main__":
    fw = open("../src_tgt_process_1.txt",mode="w",encoding="utf-8")
    num = 0
    num_fail = 0
    with open("../src_tgt.txt", mode="r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line != "":
                try:
                    sen1, sen2 = line.split("---xhm---")
                    res = isParaphrase_pair(sen1, sen2)
                    if res == True:
                        fw.write(line+"\n")
                    # print(sen1,"#######",sen2,"####",res)
                    # s=input("继续输入y,")
                    # while s!="y":
                    #     s = input("继续输入y,")
                except Exception:
                    print("出错")
                    num_fail+=1
            num += 1
            if num%10000==0:
                print("---------数据清洗中--------------",num/233864191)
    print("num", num)
    print("出错的个数：", num_fail)
    fw.close()
