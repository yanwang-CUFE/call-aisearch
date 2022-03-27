# from getinitMatrix2 import json_to_dict
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

def read_matrix_data(pt_path):
    y = torch.load(pt_path)
    return y


class BertAISearchModel4(nn.Module):

    def __init__(self):
        super(BertAISearchModel4, self).__init__()
        self.bert = AlbertModel.from_pretrained('clue/albert_chinese_tiny')

    def forward(self, ids, mask):
        _, pooled_output = self.bert(
            input_ids=ids,
            attention_mask=mask,
            return_dict=False
        )
        return pooled_output


class SimilarDocumentModel4():
    """
    call model return json
    """
    def __init__(self, model, tokenizer, all_matrix_document, all_matrix_query, device, query_dict):
        """
        all_matrix  5000 * 1 * 312
        :param model:
        :param tokenizer:
        :param all_matrix:
        """
        self.model = model
        self.tokenizer = tokenizer
        self.allMatrixDocument = all_matrix_document
        self.allMatrixQuery = all_matrix_query
        self.device = device
        self.query_dict = query_dict

# 打印某个问题在原始中的向量
    def get_matrix_vector(self, index, type_):
        print(self.query_dict[index])
        if type_ == "document":
            print(self.allMatrixDocument[int(index)])
            return str(self.allMatrixDocument[int(index)])
        else:
            print(self.allMatrixQuery[int(index)])
            return str(self.allMatrixQuery[int(index)])

    def get_one_vector(self, query):
        encoding = self.tokenizer(
            str(query),
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        # input_id = encoding['input_ids'].to(self.device)
        # attention_mask = encoding['attention_mask'].to(self.device)
        input_id = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        self.model.eval()
        with torch.no_grad():
            vector = self.model(input_id, attention_mask)
        return vector

# 和document 比较
    def get_similar_matrix1(self, vector):
        """
        :param vector:one doc vector
        :return: simi 1 * n(number of document)
        """
        simi = F.cosine_similarity(vector.unsqueeze(1), self.allMatrixDocument.unsqueeze(0), dim=2)
        return simi

    # 和query比较
    def get_similar_matrix2(self, vector):
        """
        :param vector:one doc vector
        :return: simi 1 * n(number of document)
        """
        simi = F.cosine_similarity(vector.unsqueeze(1), self.allMatrixQuery.unsqueeze(0), dim=2)
        return simi

    def get_top_1(self, simi_matrix):
        col = simi_matrix.size()[1]
        max_index = 0
        max_cos = simi_matrix[0][0]
        dict_return = {}
        for i in range(int(col)):
            # print(simi_matrix[0][i])
            if (simi_matrix[0][i] > max_cos):
                print("-----")
                print(i)
                max_index = i
                max_cos = simi_matrix[0][i]
                print(max_cos)
                print(self.query_dict[str(max_index)])

                dict_return = self.query_dict[str(max_index)]
        print(max_index)
        # dict_return = self.query_dict[str(max_index)]
        print(dict_return)
        # data_dict_json = json.dumps(dict_return, ensure_ascii=False)
        # return data_dict_json
        return max_cos, dict_return


    def return_json(self, simi_matrix):
        """
        返回固定格式的json
        :param simi_matrix: 相似度矩阵
        :return: zq baby 要求的json数据
        """
        sort, indices = torch.sort(simi_matrix, descending=True, dim=1)
        _, rank = indices.sort(dim=1)  # 再对索引升序排列，得到其索引作为排名rank
        data_dict = {"totalHits": rank.size()[1]}
        indices0 = rank[0]
        print(indices0)
        scoreDocs = []


        # # dict save data
        # score_dict = {}
        for index, i in enumerate(indices0):
            scoreDocs.append({"doc": index, "score": int(i.detach())})
        data_dict["scoreDocs"] = scoreDocs
        data_dict_json = json.dumps(data_dict, ensure_ascii=False)
        return data_dict_json
        #
        # # np save data
        # # sort_list = np.zeros(1, simi_matrix.size()[0])
        # # for index, i in indices0:
        # #     sort_list[i] = index
        #
        # data_dict["ScoreDots"] = score_dict

    def process1(self, query):
        query_vector = self.get_one_vector(query)
        print(query_vector.size())
        print(self.allMatrixDocument.size())
        simi_matrix = self.get_similar_matrix1(query_vector)
        # print(simi_matrix)
        # simi_json = self.get_top_1(simi_matrix)
        # return simi_json

        max_cosin, simi_json = self.get_top_1(simi_matrix)
        return max_cosin, simi_json

    def process2(self, query):
        query_vector = self.get_one_vector(query)
        print(query_vector.size())
        print(self.allMatrixQuery.size())
        simi_matrix = self.get_similar_matrix2(query_vector)
        # print(simi_matrix)
        # simi_json = self.get_top_1(simi_matrix)
        # return simi_json
        max_cosin, simi_json = self.get_top_1(simi_matrix)
        return max_cosin, simi_json

    def process3(self, query, index, type_):
        vector_init = self.get_matrix_vector(index, type_)
        vector_process = str(self.get_one_vector(query))
        return vector_init, vector_process


if __name__ == "__main__":
    # torch.load(model_path)
    pt_path = "./newMatrix/document92618matrix3.pt"

    stored_matrix = read_matrix_data(pt_path)
    print(stored_matrix[0])
    print(stored_matrix[1])



    # model_path = "./model_save2/similarQuery.bin"
    # bert_model = 'clue/albert_chinese_tiny'
    # pt_path = "./newMatrix/document92618matrix2.pt"
    # tokenizer = BertTokenizer.from_pretrained(bert_model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # # print("device=", device)
    # # # if torch.cuda.device_count() > 1:
    # # #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
    # # #     # gpu_ids = [1, 0]
    # # #     model = nn.DataParallel(model)
    # # # model.to(device)
    # model = BertAISearchModel2()
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    #
    # # model.load_state_dict(torch.load(model_path))
    # stored_matrix = read_matrix_data(pt_path)
    # query_dict = json_to_dict("combine_all.json")
    #
    # model_demo = SimilarQueryModel(model, tokenizer, stored_matrix, device, query_dict)
    # test_query = "国产电影成片审查办理依据是什么"
    # doc = model_demo.process(test_query)
    # print(doc)