import torch
from torch import nn, optim
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import json
import torch.nn.functional as F
from CallingDemo import BertAISearchModel
from dataUtil import get_all_divided_part
"""
将已有的文档变成向量
"""
seed = 172
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def dict_to_json(dict_temp, file_name):
    print("开始写入文件")
    print(file_name)
    with open(file_name, "w", encoding='utf-8') as f:
        json.dump(dict_temp, f, indent=2, ensure_ascii=False)
    print("结束写入文件")


def json_to_dict(file_path):
    print("开始读取文件")
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    print("结束读取文件")


all_number = 5543
# def read_xml(file_path):
#     data_list = []
#     for i in range(all_number + 1):
#         file_name = file_path + str(i) + ".xml"
#         data_list.append(get_all_divided_part(file_name))
#     return data_list


def read_json(file_path):
    dict_all = json_to_dict(file_path)
    num_len = len(dict_all)
    print(num_len)
    data_list = []
    for i in dict_all:
        # title
        # data_list.append(dict_all[i]["title"] + dict_all[i]["fulltext"] + dict_all[i]["standard"])
        data_list.append(dict_all[i]["title"])

    print(len(data_list))
    print(data_list[0])
    print("lala")
    print(data_list[1])
    return data_list


model_path = "./model_save2/model7.bin"
# torch.load(model_path)
bert_model = 'clue/albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertAISearchModel()
model.load_state_dict(torch.load(model_path))
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device=", device)
if torch.cuda.device_count() > 1:
    print("Let's use ", torch.cuda.device_count(), "GPUs!")
    # gpu_ids = [1, 0]
    model = nn.DataParallel(model)
model.to(device)

# 相当于用''代替'module.'。
# 直接使得需要的键名等于期望的键名。
# 5543
model.eval()
torch_matrix = torch.zeros(all_number, 312)
xml_list = read_json("zq_2_22.json")
with torch.no_grad():
    for index, i in tqdm(enumerate(xml_list)):
        str_i = i
        encoding = tokenizer(
                str(str_i),
            # title
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        input_id = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        vector = model(input_id, attention_mask)
        torch_matrix[index] = vector
torch.save(torch_matrix, "./newMatrix/ZQ0222document5543matrixTitle.pt")

# y = torch.load("./myTensor.pt")
# print(y)
