from NNet import NNetWrapper as nn
from GobangGame import GobangGame as Game
from Arena import Arena
import numpy as np
from MCTS import MCTS
from utils import *
from GobangGame import display, display_pi
from GobangLogic import Board
from multiprocessing import Process, Queue, Pool


args = dotdict({
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 25,
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


def arena_process(i):
    g = Game(8)

    nnet = nn(g)
    nnet.load_model(filename="third_model")
    nmcts = MCTS(g, nnet, args)

    pnet = nn(g)
    pnet.load_model(filename="second_model")
    pmcts = MCTS(g, pnet, args)

    def player1(x):
        pi = pmcts.get_action_prob(x)
        # display_pi(np.array(pi[:-1]).reshape((len(x), len(x))))
        return np.random.choice(len(pi), p=pi)

    def player2(x):
        pi = nmcts.get_action_prob(x)
        return np.random.choice(len(pi), p=pi)

    arena = Arena(player1=lambda x: player1(x), player2=lambda x: player2(x), game=g, display=display)
    return arena.play_games(8)


def f(x):
    return x * x


if __name__ == '__main__':
    with Pool(8) as p:
        result = p.map(arena_process, range(8))

    win_1 = sum([i[0] for i in result])
    win_2 = sum([i[1] for i in result])

    print(win_1)
    print(win_2)



