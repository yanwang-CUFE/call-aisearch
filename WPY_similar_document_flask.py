import flask
from flask import Flask, request, Response
import json
from WPY_similar_document_utils import SimilarDocumentModel4, BertAISearchModel4, read_matrix_data, json_to_dict
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# 初始化一个类
model_path = "./model_save2/similarQuery2.bin"

bert_model = 'clue/albert_chinese_tiny'
pt_path_document = "./newMatrix/WPYNewdocument92618matrixDocument.pt"
pt_path_query = "./newMatrix/WPYNewdocument92618matrixQuery.pt"

tokenizer = BertTokenizer.from_pretrained(bert_model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # print("device=", device)
# # if torch.cuda.device_count() > 1:
# #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
# #     # gpu_ids = [1, 0]
# #     model = nn.DataParallel(model)
# # model.to(device)
model = BertAISearchModel4()
model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})

# model.load_state_dict(torch.load(model_path))
stored_matrix_docyment = read_matrix_data(pt_path_document)
stored_matrix_query = read_matrix_data(pt_path_query)

query_dict = json_to_dict("combine_all.json")

model_demo = SimilarDocumentModel4(model, tokenizer, stored_matrix_docyment, stored_matrix_query, device, query_dict)
# test_query = "如何办理驾照"
# doc = model_demo.process(test_query)
# print(doc)
#
# SimilarQueryModelDemo =


# 结巴分词



# http://127.0.0.1:5000/similarQuery?query=如何办理驾照

# @app.route('/similarDocument', methods=['GET'])
# def search1():
#     text = request.args["query"]
#     rt = model_demo.process(text)
#     return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')


@app.route('/FAQ', methods=['GET'])
def search1():
    text = request.args["First_utterance"]
    Service_name = request.args["Service_name"]
    search_way = request.args["way"]
    if search_way == "title":
        max_cosin, max_str = model_demo.process2(text)
    else:
        max_cosin, max_str = model_demo.process1(text)

    # print(max_cosin.numpy())
    # print(max_str["document"])
    rt = {"Similarity_score": str(max_cosin.numpy()), "answer": max_str["document"]}
    return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')

@app.route('/FAQTest', methods=['GET'])
def search2():
    index = request.args["index"]
    type_ = request.args["type_"]
    query = request.args["query"]
    vect1, vect2 = model_demo.process3(query, index, type_)
    rt = {"matrix_vector": str(vect1), "process_vector": str(vect2)}
    return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=5600)

# app.config['JSON_AS_ASCII'] = False
# app.run(host='0.0.0.0', port=5001)