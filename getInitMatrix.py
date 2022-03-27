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
from dataUtil import get_all_divided_part, get_all_title
"""
将已有的文档变成向量
"""

all_number = 5333
def read_xml(file_path):
    data_list = []
    for i in range(all_number + 1):
        file_name = file_path + str(i) + ".xml"
        data_list.append(get_all_title(file_name))
    return data_list


model_path = "./model_save2/model5.bin"
# torch.load(model_path)
bert_model = 'clue/albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertAISearchModel()
# model.load_state_dict(torch.load(model_path))
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
model.eval()
torch_matrix = torch.zeros(5334, 312)
xml_list = read_xml("./indexXML_1106_nodoc/")
with torch.no_grad():
    for index, i in tqdm(enumerate(xml_list)):
        str_i = i
        encoding = tokenizer(
                str(str_i),
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )
        input_id = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        vector = model(input_id, attention_mask)
        torch_matrix[index] = vector
torch.save(torch_matrix, "./document5334matrix4_title.pt")

# y = torch.load("./myTensor.pt")
