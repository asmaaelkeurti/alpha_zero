import pickle
from NNet import NNetWrapper as nn
from MCTS import MCTS
import numpy as np
from GobangGame import display
from Arena import Arena


class AutoRun:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    @staticmethod
    def merge_data(filename):
        file_list = [filename]
        objects = []
        train_examples = []
        for f in file_list:
            with(open(f, "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break
        for o in objects:
            train_examples += o
        return train_examples

    def arena_process(self, r, old_model_file, new_model_file, verbose=False):
        old_net = nn(self.game)
        old_net.load_model(filename=old_model_file)
        old_mcts = MCTS(self.game, old_net, self.args)

        new_net = nn(self.game)
        new_net.load_model(filename=new_model_file)
        new_mcts = MCTS(self.game, new_net, self.args)

        def old_player(x):
            pi = old_mcts.get_action_prob(x)
            # display_pi(np.array(pi[:-1]).reshape((len(x), len(x))))
            return np.random.choice(len(pi), p=pi)

        def new_player(x):
            pi = new_mcts.get_action_prob(x)
            return np.random.choice(len(pi), p=pi)

        arena = Arena(player1=lambda x: old_player(x), player2=lambda x: new_player(x), game=self.game, display=display)
        return arena.play_games(r, verbose=verbose)

    def arena_process_parallel(self, arguments):
        return self.arena_process(arguments[0], arguments[1], arguments[2], arguments[3])



