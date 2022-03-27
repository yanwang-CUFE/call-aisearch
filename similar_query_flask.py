import flask
from flask import Flask, request, Response
import json
from similar_query_utils import SimilarQueryModel, BertAISearchModel2, read_matrix_data, json_to_dict
import torch
from transformers import BertTokenizer
import random
import numpy as np

app = Flask(__name__)
seed = 172
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 初始化一个类
bert_model = 'clue/albert_chinese_tiny'
pt_path = "./newMatrix/document92618matrixQuery.pt"
tokenizer = BertTokenizer.from_pretrained(bert_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print("device=", device)
# # if torch.cuda.device_count() > 1:
# #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
# #     # gpu_ids = [1, 0]
# #     model = nn.DataParallel(model)
# # model.to(device)
model = BertAISearchModel2()
# model.load_state_dict(torch.load(model_path))
stored_matrix = read_matrix_data(pt_path)
query_dict = json_to_dict("combine_all.json")

model_demo = SimilarQueryModel(model, tokenizer, stored_matrix, device, query_dict)
# test_query = "如何办理驾照"
# doc = model_demo.process(test_query)
# print(doc)
#
# SimilarQueryModelDemo =


# 结巴分词



# http://127.0.0.1:5000/similarQuery?query=如何办理驾照
# @app.route('/similarQuery', methods=['GET'])

@app.route('/FAQ', methods=['GET'])
def search1():
    text = request.args["Service_name"]
    rt = model_demo.process(text)
    return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=5556)