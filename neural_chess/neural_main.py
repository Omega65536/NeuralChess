import chess
import torch
import badgyal
import time
import random
import custom_net
import timeit

num_nodes = 0

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

            if self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_insufficient_material():
                return

            num_moves = len(self.policy)
            threshold = 0.04
            cmoves = [move for move, prob in self.policy.items() if prob >= threshold]

            #print(cmoves)
            for move in cmoves:
                self.board.push(chess.Move.from_uci(move))
                self.children[move] = NeuralEngine.Node(self.board.copy())
                self.board.pop()

        def minimax(self, noise):
            if not self.children: return (- self.value, []) # + random.uniform(- noise, noise), [])

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
        lastpv, pv = [] , []

        while time.time() < startTime + maxTime:
            iterations += 1
            lastpv = pv
            score, pv = self.root.minimax(0.1)
            conf = len([i for i in range(min(len(pv), len(lastpv))) if pv[i] == lastpv[i]])
            print(f"ITER: {iterations:<4}\tEVAL: {score * -50 + 50: .1f}%\t" +
            f"DEPTH: {len(pv):<4}\t" + f"PV: {' '.join(pv)}")
            self.root.expand(pv)

        score, pv = self.root.minimax(0)
        print("-"*120)
        print(f"FINAL RESULT:\tEVAL: {score * -50 + 50: .1f}%\t" +
        f"DEPTH: {len(pv):<4}\t" + f"PV: {' '.join(pv)}")
        deltaTime = time.time() - startTime

        print(f"SEARCH INFO:\tNODES: {num_nodes}\tTIME: {deltaTime:.1f}s\tNPS: {num_nodes / deltaTime:.1f}")
        print("-"*120)
        print()
        return score, pv

    def output(self):
        self.root.output()

def getEval(board, net = gg_net):
    global num_nodes
    num_nodes += 1

    if board.is_checkmate():
        return {}, -100
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return {}, 0
    return net.eval(board, softmax_temp = 1.61)

if __name__ == '__main__':
    print("READY!")
    board = chess.Board()

    while True:
        inp = input()
        if inp == "exit":
            break
        elif inp == "board":
            print(board)
        elif inp == "back":
            board.pop()
        elif inp == "go":
            engine = NeuralEngine(board)
            mm = engine.getMove(10)
            board.push(chess.Move.from_uci(mm[1][0]))
            print(mm)
        elif inp == "golong":
            engine = NeuralEngine(board)
            mm = engine.getMove(30)
            print(mm)
        else:
            try:
                move = chess.Move.from_uci(inp)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("[ERROR] Illegal move")
            except:
                print("[ERROR] Invalid move")
