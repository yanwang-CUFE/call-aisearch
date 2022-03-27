# -*- coding:utf-8 -*-
import json
import os
import re
import jieba
from tqdm import tqdm
# DOM 解析
from xml.dom.minidom import parse

def dict_to_json(dict_temp, file_name):
    print("开始写入文件")
    print(file_name)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(dict_temp, f, indent=2, ensure_ascii=False)
    print("结束写入文件")


def json_to_dict(file_path):
    print("开始读取文件")
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_absolute_path_list(file_path):
    list_all = os.listdir(file_path)
    # print(list_all)
    for i_num, i in enumerate(list_all):
        list_all[i_num] = file_path + i
    # print(list_all)
    return list_all


def my_split(str_):
    return re.split('[，。：、（）《》！\n]', str_)


def get_all_divided_part(file_path):
    dom = parse(file_path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 title
    title = data.getElementsByTagName('title')[0].childNodes[0].nodeValue
    # 获取 fulltext
    fulltext = data.getElementsByTagName('fulltext')[0].childNodes[0].nodeValue
    return title + " " + fulltext


def get_all_title(file_path):
    dom = parse(file_path)
    # 获取文档元素对象
    data = dom.documentElement
    # 获取 title
    title = data.getElementsByTagName('title')[0].childNodes[0].nodeValue
    # 获取 fulltext
    fulltext = data.getElementsByTagName('fulltext')[0].childNodes[0].nodeValue
    return title


def get_3_word(list_):
    list_return = []
    for i in range(len(list_) - 3):
        list_return.append(list_[i] + list_[i + 1] + list_[i + 2])
    # print(list_return)
    return list_return


def get_4_word(list_):
    list_return = []
    for i in range(len(list_) - 4):
        list_return.append(list_[i] + list_[i + 1] + list_[i + 2] + list_[i + 3])
    # print(list_return)
    return list_return


def get_2_word(list_):
    list_return = []
    for i in range(len(list_) - 2):
        list_return.append(list_[i] + list_[i + 1])
    # print(list_return)
    return list_return


def get_all_gram(str_):
    list_all = my_split(str_)
    list_return = []
    for i in list_all:
        list_tmp = list(jieba.cut(i))
        # print(list_tmp)

        if len(list_tmp) >= 2:
            list_return += get_2_word(list_tmp)
        if len(list_tmp) >= 3:
            list_return += get_3_word(list_tmp)
        if len(list_tmp) >= 4:
            list_return += get_4_word(list_tmp)
    return list_return


def final():
    path_list = get_absolute_path_list("D:/data/pycharmWorkspace/luceneData/tianjingData/")
    dict_return = {}
    for path in tqdm(path_list):
        s1 = get_all_divided_part(path)
        list_all = get_all_gram(s1)
        for i in list_all:
            if i in dict_return:
                dict_return[i] += 1
            else:
                dict_return[i] = 1
    dict_to_json(dict_return, "final.json")


def check_top_k(file_path, k):
    num = 0
    dict_return = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line_i in lines:
            if ":" in line_i:
                head = int(line_i.split(":")[0].strip())
                bs64_str = re.findall("\(\[(.*?)\]\)", line_i)[0]
                list_t = bs64_str.split(",")
                list_temp = []
                for j in list_t:
                    list_temp.append(int(j.strip()))
                dict_return[head] = list_temp
    for i in dict_return:
        for j in range(k):
            if i == dict_return[i][j]:
                num += 1
    print(num / 128)
    return num


def get_json_from_document(file_path, all_number, json_name):
    dict_return = {}
    for i in range(all_number + 1):
        file_name = file_path + str(i) + ".xml"
        dict_return[str(i)] = get_all_title(file_name)
    dict_to_json(dict_return, json_name)
    print("end")


if __name__ == '__main__':
    print("main")
    get_json_from_document("indexXML_1106_nodoc/", 5333, "indexXML_1106_nodoc_5334_title.json")


    # check_top_k("train40/valid29.txt", 1)
    # check_top_k("train40/valid29.txt", 3)
    # check_top_k("train40/valid29.txt", 5)
    # check_top_k("train40/valid29.txt", 10)


    # dict_ = json_to_dict("valid30.json")["0"]
    # q_list = dict_["query"]
    # d_list = dict_["document"]
    # dict_return = {}
    # for i in range(len(q_list)):
    #     dict_return[i] = {"query": q_list[i], "document" : d_list[i]}
    # dict_to_json(dict_return, "valid30_all.json")


