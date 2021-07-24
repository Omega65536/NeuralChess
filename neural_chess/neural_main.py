import chess
import torch
import badgyal
import time
import random
import custom_net
import timeit

#net_10x128 = custom_net.Net10x128(cuda = False)
gg_net = badgyal.GGNet(cuda = False)
bg_net = badgyal.BGNet(cuda = False)
mg_net = badgyal.MGNet(cuda = False)

class NeuralEngine:
    class Node:
        def __init__(self, board):
            self.board = board
            self.children = {}
            self.policy, self.value = getEval(board)
            self.visits = 0

        def output(self, depth = 0):
            print(f"Visits: {self.visits} Value: {self.value * 100: .1f}% PM: {max(self.policy, key = self.policy.get)}")
            for move in self.children.keys():
                print("|   " * (depth + 1) + move + ": ", end = "")
                self.children[move].output(depth + 1)

        def expand(self, pv):
            if pv:
                move = pv[0]
                self.children[move].expand(pv[1:])
                return

            num_moves = len(self.policy)
            threshold = 0.03
            for move, p in self.policy.items():
                if p <= threshold: continue

                self.board.push(chess.Move.from_uci(move))
                self.children[move] = NeuralEngine.Node(self.board.copy())
                self.board.pop()

        def minimax(self, noise):
            if not self.children: return (- self.value + random.uniform(- noise, noise), [])

            bestValue = -1
            pv = [max(self.policy, key = self.policy.get)]
            for move, child in self.children.items():
                    mm = child.minimax(noise)
                    if mm[0] >= bestValue:
                        bestValue = mm[0]
                        pv = [move] + mm[1]
            return (-bestValue, pv)

    def __init__(self, board):
        self.board = board
        self.root = self.Node(board)
        self.num_nodes = 0

    def getMove(self, maxTime):
        startTime = time.time()
        iterations = 0
        while time.time() < startTime + maxTime:
            iterations += 1
            mm = self.root.minimax(0.1)
            print(f"ITER: {iterations:<4}\tEVAL: {mm[0]: .2f}\tPV: {' '.join([str(i) for i in mm[1]])}")
            self.root.expand(mm[1])

        mm = self.root.minimax(0)
        #self.output()
        return mm

    def output(self):
        self.root.output()

def getEval(board, net = bg_net):
    return net.eval(board, softmax_temp = 1.61)

if __name__ == '__main__':
    board = chess.Board()
    while board.legal_moves:
        engine = NeuralEngine(board)
        mm = engine.getMove(15)
        print(mm)
        board.push(chess.Move.from_uci(mm[1][0]))
        print(board)
        while True:
            try:
                board.push(chess.Move.from_uci(input()))
                break
            except:
                print("Invalid move")
                pass
    print(board)
    print()
