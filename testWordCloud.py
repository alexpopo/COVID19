# -*- coding: utf-8 -*-
"""
# @Time     : 2023/1/6 21:23
# @Author   : popo
# @File     : testWordCloud.py
# @Note     : python-v3.6.13、jieba-v0.42.1、wordcloud-v1.8.1、pillow-v8.3.2、numpy-v1.19.5
# @Software : PyCharm
"""

# 标准库
import csv
# 正则表达式
import re
# 分词功能
import jieba
# 词云
from wordcloud import WordCloud
# 图片处理
from PIL import Image
# 矩阵运算
import numpy as np
# 绘图，数据可视化
from matplotlib import pyplot as plt


def load_stop_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f_load_sw:
        return f_load_sw.read().split("\n")


# 获取训练数据
msg_lines = []
csv_file = "./data/train/institute.csv"
stopwords_file = "./data/train/stopwords.txt"

jieba.add_word('贝塞斯达', freq=1000)
jieba.add_word('病原体部', freq=1000)
jieba.add_word('疾病科', freq=1000)
jieba.add_word('传染病科', freq=1000)
jieba.add_word('临床中心', freq=1000)
jieba.add_word('国家卫生研究所', freq=1000)
jieba.suggest_freq(('贝塞斯达', '病原体部', '疾病科', '国家卫生研究所'), tune=True)

with open(csv_file) as f_csv:
    # 获取"停用词"信息
    stop_words = load_stop_words(stopwords_file)
    # print(stop_words)

    # 获取所有语料数据
    csv_reader = csv.reader(f_csv, delimiter='\t')

    # 分词
    valid_words = []
    for idx, line in enumerate(csv_reader, start=1):
        sentence = line[1].strip()
        cut_word = jieba.lcut(sentence)
        # print(idx)
        # print(cut_word)
        valid_words = [word for word in cut_word if word not in stop_words]
        # print(valid_words)
        msg_lines.append(valid_words)
        # print("-"*80)


# 保存分词过滤结果到本地
# with open('./data/train/institute_result.csv', 'x', newline='')as f_csv_save:
#     write_handle = csv.writer(f_csv_save)
#     write_handle.writerows(msg_lines)

# 词云输入中文文本
symbol = " "
wc_text = ""
for idx in range(len(msg_lines)):
    temp_text = symbol.join(msg_lines[idx])
    if len(wc_text) == 0:
        wc_text += temp_text
    else:
        wc_text += ' ' + temp_text
# print("+"*80)
# print(wc_text)

img = Image.open(r".\data\img\Lung.jpeg")
img_array = np.array(img)   # 将图片转换为数组

wc = WordCloud(
    font_path="msyh.ttc",
    background_color='white',
    mask=img_array,
    max_font_size=100,
    collocations=False
)

wc.generate_from_text(wc_text)

# 绘制图片
fig = plt.figure(1)
plt.imshow(wc)
plt.axis('off')     # 设置是否显示坐标轴

# 输出词云图片到文件
plt.savefig(r".\data\img\lung_wordcloud.jpeg", dpi=200)

plt.show()
print("create word cloud finish.")
