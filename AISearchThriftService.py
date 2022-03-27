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
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket, TTransport
from SearchService import AISearchService
from CallingDemo import BertAISearchModel, read_matrix_data, CallModel, CallModel2


class SearchThriftServer(AISearchService.Iface):
    def __init__(self):
        self.model_path = "./model_save2/model7.bin"
        # torch.load(model_path)
        self.bert_model = 'clue/albert_chinese_tiny'
        self.pt_path = "./newMatrix/ZQ0222document5543matrix.pt"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print("device=", device)
        # if torch.cuda.device_count() > 1:
        #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
        #     # gpu_ids = [1, 0]
        #     model = nn.DataParallel(model)
        # model.to(device)
        self.model = BertAISearchModel()
        # model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(self.model_path).items()})
        self.stored_matrix = read_matrix_data(self.pt_path)
        self.pt_path2 = "./newMatrix/ZQ0222document5543matrixTitle.pt"

        self.stored_matrix2 = read_matrix_data(self.pt_path2)

        # self.model_demo = CallModel(self.model, self.tokenizer, self.stored_matrix, self.device)
#       title
        # torch.load(model_path)
        # self.pt_path2 = "./document5334matrix4_title.pt"


        self.model_demo = CallModel2(self.model, self.tokenizer, self.stored_matrix, self.stored_matrix2, self.device)

    def AISearch(self, query, Ranking, Title):
        if Ranking:
            if Title:
                json_str = self.model_demo.process2(query)
            else:
                json_str = self.model_demo.process(query)
        else:
            if Title:
                json_str = self.model_demo.process4(query)
            else:
                json_str = self.model_demo.process3(query)
        return json_str


if __name__ == "__main__":
    # model_path = "./model1/model.bin"
    # # torch.load(model_path)
    # bert_model = 'clue/albert_chinese_tiny'
    # pt_path = "./document5333matrix.pt"
    # tokenizer = BertTokenizer.from_pretrained(bert_model)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # print("device=", device)
    # # if torch.cuda.device_count() > 1:
    # #     print("Let's use ", torch.cuda.device_count(), "GPUs!")
    # #     # gpu_ids = [1, 0]
    # #     model = nn.DataParallel(model)
    # # model.to(device)
    # model = BertAISearchModel()
    # # model.load_state_dict(torch.load(model_path))
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    # stored_matrix = read_matrix_data(pt_path)
    #
    # model_demo = CallModel(model, tokenizer, stored_matrix, device)
    doc = "放射性同位素与射线装置豁免备案"
    # json_str = model_demo.process(doc)
    #
    # print("end")
    # # simi_matrix = torch.tensor([[1.3, 3.5, 1.2, 5]])
    # print(json_str)

    __HOST = '10.13.56.36'
    __PORT = 9092
    handler = SearchThriftServer()
    processor = AISearchService.Processor(handler)
    transport = TSocket.TServerSocket(host=__HOST, port=__PORT)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    print('Starting the server')
    server.serve()
    print('done')