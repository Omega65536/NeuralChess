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
        FILE = "custom_nets/weights_run2_744701.pb.gz"
        FILTERS = 128
        BLOCKS = 10
        SE = 2

        net = model.Net(FILTERS, BLOCKS, FILTERS, SE, classical=True)
        net.import_proto_classical(FILE)
        return net
