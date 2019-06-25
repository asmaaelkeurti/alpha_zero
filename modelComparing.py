from NNet import NNetWrapper as nn
from GobangGame import GobangGame as Game
from Arena import Arena
import numpy as np
from MCTS import MCTS
from utils import *
from GobangGame import display, display_pi
from GobangLogic import Board


args = dotdict({
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 25,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 1000,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


if __name__ == '__main__':
    g = Game(8)

    pnet = nn(g)
    nnet = nn(g)
    nnet.load_model(filename="second_model")

    pmcts = MCTS(g, pnet, args)
    nmcts = MCTS(g, nnet, args)

    def player1(x):
        pi = pmcts.get_action_prob(x)
        # display_pi(np.array(pi[:-1]).reshape((len(x), len(x))))
        return np.random.choice(len(pi), p=pi)

    def player2(x):
        pi = nmcts.get_action_prob(x)
        return np.random.choice(len(pi), p=pi)


    # b = Board(8)
    # canonicalBoard = g.get_canonical_form(np.array(b.pieces), 1)

    # player1(canonicalBoard)

    arena = Arena(player1=lambda x: player1(x), player2=lambda x: player2(x), game=g, display=display)

    arena.play_game(verbose=True)
