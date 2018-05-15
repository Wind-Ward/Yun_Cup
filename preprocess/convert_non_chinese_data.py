import pandas as pd
import re



train_22w=pd.read_csv("./data/train_22w_non_Chinese_654.csv")
test_5w=pd.read_csv("./data/test_5w_non_Chinese_137.csv")
test_3w=pd.read_csv("./data/test_a_3w_non_Chinese_90.csv")
## 上述文件通过正则匹配将非中文的评论提取出来


#对数据集中非中文的评论做正则匹配，将字符转化成中文
def transfer(text):
    temp=text.lower()
    temp= re.sub('^ddd+','非常好',temp)
    temp=re.sub("^hhh+","非常好",temp)
    temp=re.sub("0k","好",temp)
    temp = re.sub("〇k", "好", temp)
    temp = re.sub("ok", "好", temp)
    temp = re.sub("hen", "很", temp)
    temp = re.sub("bucuo", "不错", temp)
    temp = re.sub("zan", "赞", temp)
    temp=re.sub("feichang","非常",temp)
    temp=re.sub("^66+","非常好",temp)
    temp = re.sub("hao", "好", temp)
    temp = re.sub("^23+", "非常好", temp)
    temp = re.sub("hehe", "呵呵", temp)
    temp=re.sub("cool","很酷",temp)
    temp=re.sub("^_^","开心",temp)
    temp=re.sub("yi ban","一般",temp)
    temp = re.sub("yiban", "一般", temp)
    temp = re.sub("nice", "不错", temp)
    temp = re.sub("good", "好", temp)
    temp = re.sub("happy", "快乐", temp)
    temp = re.sub("kaixin", "开心", temp)
    temp = re.sub("youwan", "游玩", temp)

    return temp


train_22w["discuss_translate"]=test_5w["Discuss"].apply(transfer)
train_22w.to_csv("./output/test_5w_non_Chinese_translate_137.csv",index=False)

test_5w["discuss_translate"]=test_5w["Discuss"].apply(transfer)
test_5w.to_csv("./output/test_5w_non_Chinese_translate_137.csv",index=False)

test_3w["discuss_translate"]=test_3w["Discuss"].apply(transfer)
test_3w.to_csv("./output/test_3w_non_Chinese_translate_90.csv",index=False)


#再将上述生成好的文件在将未匹配的英文手工翻译成中文