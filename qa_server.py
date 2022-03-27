# import flask
# import json
# from transformers import AutoModelForQuestionAnswering,AutoTokenizer,pipeline
# import json
# from collections import Counter
# import requests
# import sys
# import time
# sys.stdout = open(1, 'w', encoding='utf-8', closefd=False)
#
# server=flask.Flask(__name__)
#
# @server.route('/qa',methods=['post','get'])
# def qa_server():
#     context=flask.request.values.get('context')
#     question=flask.request.values.get('question')
#     qa_res = QA({'context':context,"question":question})
#     return json.dumps(qa_res,ensure_ascii=False)
#
# model = AutoModelForQuestionAnswering.from_pretrained('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large')
# tokenizer = AutoTokenizer.from_pretrained('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large')
# QA = pipeline('question-answering', model=model, tokenizer=tokenizer)
#
# server.run(port=5837)