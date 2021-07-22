import chess
import torch
import badgyal
import time
import random
import main_net
import timeit

tg_net = main_net.MainNet(cuda = False)
gg_net = badgyal.GGNet(cuda = False)
bg_net = badgyal.BGNet(cuda = False)
mg_net = badgyal.MGNet(cuda = False)

class Node:
    def __init__(self, board):
        self.board = board
        self.visits = 0
        self.children = {}
        self.policy, self.value = getEval(board)
        #print(f"NEW NODE: {board.fen()}")

    def expand(self):
        if self.children:
            delta = 0.1
            move = max(self.policy, key = lambda x: self.policy[x] + random.uniform(-delta, delta))
            self.visits += 1
            self.children[move].expand()
            return

        num_moves = len(self.policy)
        for move, p in self.policy.items():
            if p <= 1 / num_moves: continue

            if move in self.children:
                self.children[move].expand(depth + 1)
            else:
                self.board.push(chess.Move.from_uci(move))
                self.children[move] = Node(self.board.copy())
                self.board.pop()

    # def expand(self, depth):
    #     delta = max((1 / (depth + 1)) * 0.1, 0.1)
    #     move = max(self.policy, key = self.policy.get)
    #     move = max(self.policy,
    #         key = lambda x: self.policy[x] + random.uniform(-delta, delta))
    #     self.visits += 1
    #
    #     if move in self.children:
    #         self.children[move].expand(depth + 1)
    #     else:
    #         self.board.push(chess.Move.from_uci(move))
    #         self.children[move] = Node(self.board.copy())
    #         self.board.pop()
    #
    #     #print(f"EXPANDING: {board.fen()}; MOVE: {move}")

    def output(self, depth = 0):
        if depth == 0: print(f"ROOT: ", end = "")
        print(f"Visits: {self.visits} Value: {self.value * 100: .1f}% PM: {max(self.policy, key = self.policy.get)}")
        for move, child in self.children.items():
            print(".   " * (depth + 1) + move + ": ", end = "")
            child.output(depth + 1)

    def minimax(self):
        if not self.children:
            return (- self.value, [])

        bestValue = -1
        pv = [max(self.policy, key = self.policy.get)]
        for move, child in self.children.items():
            mm = child.minimax()
            if mm[0] >= bestValue:
                bestValue = mm[0]
                pv = [move] + mm[1]
        return (-bestValue, pv)



def getEval(board, net = gg_net):
    policy, value = net.eval(board, softmax_temp = 1.61)
    #print(f"NET: {type(net)}; VALUE: {value * 100: .1f}%; BEST POLICY: {max(policy, key = policy.get)}")
    return policy, value

def getMove(board):
    root = Node(board)
    for i in range(10):
        root.expand()
    root.output()
    mm = root.minimax()
    print(mm)

    return chess.Move.from_uci(mm[1][0])

BOARDS = [
    chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPPKPPP/RNBQ1BNR b kq - 0 1"),
    chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    chess.Board("rn1qkbnr/pp3ppp/4p3/2ppP3/3P4/3Q1N2/PPP2PPP/RNB1K2R w KQkq - 0 1"),
    chess.Board("r1b1k1nr/pp5p/n2qppp1/3pN2Q/2pP4/2P1P3/PP3PPP/RN2KB1R w KQkq - 0 10"),
    chess.Board("4k3/8/8/8/8/8/8/3RK3 b - - 0 1")
]

if __name__ == '__main__':
    board = chess.Board()
    while board.legal_moves:
        move = getMove(board)
        board.push(move)
        print(move)
        board.push(chess.Move.from_uci(input()))
    print(board)
    getMove(board)
    print()
