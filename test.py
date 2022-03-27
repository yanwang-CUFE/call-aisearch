# coding:utf-8

# import torch
# from torch import nn, optim
# from torch.optim import AdamW
# from tqdm import tqdm
# from transformers import BertTokenizer, BertModel
# from transformers import AlbertConfig, AlbertModel, AlbertTokenizer
# from torch.utils.data import Dataset, DataLoader, random_split
# import numpy as np
# import random
# import json
# import torch.nn.functional as F

"""
用来测试代码
"""

# a = torch.rand(5, 4)
#
# b = torch.rand(1, 4)
#
# matrix1 = F.cosine_similarity(b, a)
# matrix2 = F.cosine_similarity(a, b)
# print(a)
# print(b)
# print(matrix1)
# print(matrix2)

# a = torch.tensor([[1.1, 2, 3, 4]])
# b = torch.tensor([[1.1, 2, 3, 4], [4, 3, 2, 1]])
# logits = torch.tensor([[-0.5816, -0.3873, -1.0215, -1.0145, 0.4053],
#                         [0.7265, 1.4164, 1.3443, 1.2035, 1.8823],
#                         [-0.4451, 0.1673, 1.2590, -2.0757, 1.7255],
#                         [0.2021, 0.3041, 0.1383, 0.3849, -1.6311]])
#
# simi_matrix = torch.tensor([[1.3, 3.5, 1.2, 5]])
#
# sorted_logits, sorted_indices = torch.sort(simi_matrix, descending=True, dim=1)  # 对logits进行递减排序
# print(sorted_logits)
# print(sorted_indices)
# _, rank = sorted_indices.sort(dim=1)  # 再对索引升序排列，得到其索引作为排名rank
# print(rank)


# print(a.size())
# print(b.size())
# # a = torch.rand((4, 64))
# # b = torch.rand((5, 64))
# simi = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
# print(simi)
import json

def json_to_dict(file_path):
    print("开始读取文件")
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    print("结束读取文件")
# doc = "['此事项为非依申请的公共服务，无需申请。', '行政审批部门根据利害关系人的请求或者依据职权，对下列行为实施撤销行政许可，告知行政相对人并签发《撤销行政许可决定书》：a) 对行政机关工作人员滥用职权、玩忽职守作出准予行政许可决定的；b) 超越法定职权作出准予行政许可决定的；c) 违反法定程序作出准予行政许可决定的；d) 对不具备申请资格或者不符合法定条件的申请人准予行政许可的；e) 对监管部门依法作出吊销行政许可证照处罚等情形的。', '行政审批部门根据利害关系人的请求或者依据职权，对下行为列实施撤销行政许可，告知行政相对人并签发《撤销行政许可决定书》：a)对行政机关工作人员滥用职权、玩忽职守作出准予行政许可决定的；b)超越法定职权作出准予行政许可决定的；c)违反法定程序作出准予行政许可决定的；d)对不具备申请资格或者不符合法定条件的申请人准予行政许可的；e)对监管部门依法作出吊销行政许可证照处罚等情形的。', '解答：到所在区体育部门咨询，由其组织参加培训或活动，符合要求后可申请三级社会体育指导员，并根据规定逐级晋升。', '行政审批部门根据利害关系人的请求或者依据职权，对下行为列实施撤销行政许可，告知行政相对人并签发《撤销行政许可决定书》：a)对行政机关工作人员滥用职权、玩忽职守作出准予行政许可决定的；b)超越法定职权作出准予行政许可决定的；c)违反法定程序作出准予行政许可决定的；d)对不具备申请资格或者不符合法定条件的申请人准予行政许可的；对监管部门依法作出吊销行政许可证照处罚等情形的。', '此事项为非依申请的公共服务，无需申请。', '根据《中华人民共和国档案法实施办法》第二十二条规定，档案的公布是指首次向社会公开档案的全部或者部分原文，或者档案记载的特定内容。', '行政审批部门根据利害关系人的请求或者依据职权，对下行为列实施撤销行政许可，告知行政相对人并签发《撤销行政许可决定书》：a)对行政机关工作人员滥用职权、玩忽职守作出准予行政许可决定的；b)超越法定职权作出准予行政许可决定的；c)违反法定程序作出准予行政许可决定的；d)对不具备申请资格或者不符合法定条件的申请"
# print(doc)
dict_all = json_to_dict("combine_all.json")
print(dict_all["20533"])
import torch



