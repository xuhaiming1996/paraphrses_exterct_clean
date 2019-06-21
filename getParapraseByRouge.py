from rouge import Rouge



rouge = Rouge()


def get_rouge_score(sentence_a, sentence_b):
    weight = [0.33, 0.33, 0.33]  # 分别代表rouge-1， rouge-2， rouge-l所占比重。


    sentence_a = " ".join([word for word in sentence_a])
    sentence_b = " ".join([word for word in sentence_b])

    try:
        scores = rouge.get_scores(sentence_a, sentence_b, avg=True)
    except Exception as e:
        # print(e)
        return 0
    rouge_1 = scores["rouge-1"]
    rouge_2 = scores["rouge-2"]
    rouge_l = scores["rouge-l"]
    f = rouge_1["f"] * weight[0] + rouge_2["f"] * weight[0] + rouge_l["f"] * weight[0]
    return f


def find_paraphrase(file_a_name,file_b_name,file_save_name,limit_score):
    f_a=open(file_a_name,mode="r",encoding="utf-8")
    f_b=open(file_b_name,mode="r",encoding="utf-8")
    f_w=open(file_save_name,mode="w",encoding="utf-8")
    num=0
    for sentence_a,sentence_b in zip(f_a,f_b):
        sentence_a=sentence_a.replace(" ", "").replace("\n", "").replace("\r\n", "").replace("\t", "").replace("　　", "")
        sentence_b=sentence_b.replace(" ", "").replace("\n", "").replace("\r\n", "").replace("\t", "").replace("　　", "")
        f = get_rouge_score(sentence_a, sentence_b)

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
    file_save_name = "paraphrase_pair.txt"
    find_paraphrase(file_a_name, file_b_name, file_save_name, 0.5)


