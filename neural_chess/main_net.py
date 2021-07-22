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

CHANNELS = 16
BLOCKS = 2
SE = 1

class MainNet(AbstractNet):
    def __init__(self, cuda=True, torchScript=False):
        super().__init__(cuda=cuda, torchScript=torchScript)

    def load_net(self):
        FILE = "tinygyal-8.pb.gz"
        net = model.Net(CHANNELS, BLOCKS, CHANNELS, SE, classical=True)
        net.import_proto_classical(FILE)
        return net
