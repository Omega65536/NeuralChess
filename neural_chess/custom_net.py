import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys
import os.path
from badgyal import AbstractNet

class Net10x128(AbstractNet):
    def __init__(self, cuda=True, torchScript=False):
        super().__init__(cuda=cuda, torchScript=torchScript)

    def load_net(self):
        FILE = "custom_nets/LS15-20x256SE-jj-9-75000000.pb.gz"
        FILTERS = 256
        BLOCKS = 20
        SE = 16
        net = model.Net(FILTERS, BLOCKS, FILTERS, SE, classical=True)
        net.import_proto_classical(FILE)
        return net
