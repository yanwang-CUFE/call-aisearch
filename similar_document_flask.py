import flask
from flask import Flask, request, Response
import json
from similar_document_util import SimilarDocumentModel, BertAISearchModel3, read_matrix_data, json_to_dict
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# 初始化一个类
# model_path = "./model_save2/model7.bin"
bert_model = 'clue/albert_chinese_tiny'
pt_path = "./newMatrix/document92618matrixDocument.pt"
tokenizer = BertTokenizer.from_pretrained(bert_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print("device=", device)
# # if torch.cuda.device_count() > 1:
# #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
# #     # gpu_ids = [1, 0]
# #     model = nn.DataParallel(model)
# # model.to(device)
model = BertAISearchModel3()
# model.load_state_dict(torch.load(model_path))
stored_matrix = read_matrix_data(pt_path)
query_dict = json_to_dict("combine_all.json")

model_demo = SimilarDocumentModel(model, tokenizer, stored_matrix, device, query_dict)
# test_query = "如何办理驾照"
# doc = model_demo.process(test_query)
# print(doc)
#
# SimilarQueryModelDemo =


# 结巴分词



# http://127.0.0.1:5600/FAQ?First_utterance=如何办理驾照&Service_name=政府

@app.route('/FAQ', methods=['GET'])
def search1():
    text = request.args["First_utterance"]
    Service_name = request.args["Service_name"]
    max_cosin, max_str = model_demo.process(text)
    print("ppp")
    print(max_cosin.numpy())
    print(max_str["document"])
    rt = {"Similarity_score": str(max_cosin.numpy()), "answer": max_str["document"]}
    return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=5600)