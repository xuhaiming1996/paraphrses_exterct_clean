# 简单介绍
代码充分考虑了千万甚至亿级别语料的效率性。
代码使用以下技术：bloomfilter, rouge,bert,bertService,fastText,
                   laser, faiss,pytorch，bert-as-service等等 加快处理速度。
                   
同时代码书写秉承 规范，整洁
         
               
# 描述
## 基于规则的
     ### 长度
            长度差距太大---阈值设定

     ### 算法规则
            EQUAL:是否相等---相等去掉
            最长公共子串---占百分比高低都要设置
            rouge

##   基于深度学习的
      使用基于FastText的方式计算两个句子的语义相似度--单词级别
      使用基于BERT的方式计算两个句子的语义----这里不训练bert直接拿过来用，仅仅是用来获取具有上下文的单词向量
##  基于LASER的代码
        使用laser获取句子的语义向量
    
## 使用faiss 方法在千万级别（2700万）中mini复述对,查找时间降到0.193s 一个 
具体请看代码      
            
     
    
