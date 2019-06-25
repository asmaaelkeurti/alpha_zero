from Coach import Coach
from GobangGame import GobangGame as Game
from GobangGame import display, display_pi
from NNet import NNetWrapper as nn
from utils import *
from MCTS import MCTS
from GobangLogic import Board
import pickle
import numpy as np
import os
from multiprocessing import Process, Lock


args = dotdict({
    'goBang_n': 8,
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 30,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def mcts_test():
    g = Game(8)
    b = Board(8)
    nnet = nn(g)

    mcts = MCTS(g, nnet, args)

    b.execute_move((1, 1), 1)
    b.execute_move((1, 2), 1)
    b.execute_move((1, 3), 1)
    # b.execute_move((1, 4), 1)

    b.execute_move((3, 2), -1)
    b.execute_move((3, 3), -1)
    b.execute_move((3, 4), -1)
    # b.execute_move((3, 5), -1)

    curPlayer = 1
    canonicalBoard = g.get_canonical_form(np.array(b.pieces), curPlayer)

    pi = mcts.get_action_prob(canonicalBoard)

    display(canonicalBoard)
    display_pi(np.array(pi[:-1]).reshape((len(canonicalBoard), len(canonicalBoard))))


def generate_data(l):
    g = Game(args.goBang_n)
    nnet = nn(g)

    c = Coach(g, nnet, args)
    train_example = c.execute_episode()

    l.acquire()
    try:
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder + "train_examples_3")
        with open(filename, "ab+") as f:
            pickle.dump(train_example, f)
    finally:
        l.release()


if __name__ == "__main__":
    lock = Lock()

    for iteration in range(100):
        jobs = []

        for i in range(6):
            p = Process(target=generate_data, args=(lock,))
            jobs.append(p)
            p.start()

        for job in jobs:
            job.join()

        print(iteration)

