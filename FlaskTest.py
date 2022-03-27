import flask
from flask import Flask, request, Response
import json
from similar_document_util import SimilarDocumentModel, BertAISearchModel3, read_matrix_data, json_to_dict
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# http://127.0.0.1:5005/zmyTest?query=如何办理驾照&category=政府

@app.route('/zmyTest', methods=['GET'])
def search1():
    text = request.args["query"]
    category_ = request.args["category"]
    rt = text + category_
    return Response(json.dumps(rt, ensure_ascii=False), mimetype='application/json')


app.config['JSON_AS_ASCII'] = False
app.run(host='0.0.0.0', port=5005)