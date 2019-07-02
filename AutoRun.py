import pickle
from NNet import NNetWrapper as nn
from MCTS import MCTS
import numpy as np
from GobangGame import display
from Arena import Arena
from multiprocessing import Pool, Lock, Process
from Coach import Coach
import os


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

    def arena_process_parallel_function(self, arguments):
        return self.arena_process(arguments[0], arguments[1], arguments[2], arguments[3])

    def arena_process_parallel(self, r, player1_model, palyer2_model):
        with Pool(8) as p:
            arena_result = p.map(self.arena_process_parallel_function,
                                 [[r, player1_model, palyer2_model, False] for _ in range(8)])
        return sum([i[0] for i in arena_result]), sum([i[1] for i in arena_result])

    def generate_data(self, l, model_file, train_example_filename):
        nnet = nn(self.game)
        nnet.load_model(filename=model_file)

        c = Coach(self.game, nnet, self.args)
        train_example = c.execute_episode()

        l.acquire()
        try:
            folder = self.args['checkpoint']
            if not os.path.exists(folder):
                os.makedirs(folder)
            filename = os.path.join(folder + train_example_filename)
            with open(filename, "ab+") as f:
                pickle.dump(train_example, f)
        finally:
            l.release()

    def generate_data_parallel(self, r, model_file, train_example_filename):
        lock = Lock()
        for iteration in range(r):
            jobs = []

            for _ in range(8):
                p = Process(target=self.generate_data, args=(lock, model_file, train_example_filename))
                jobs.append(p)
                p.start()

            for job in jobs:
                job.join()

    def generate_data_debug(self, model_file):
        nnet = nn(self.game)
        nnet.load_model(filename=model_file)

        c = Coach(self.game, nnet, self.args)
        train_example = c.execute_episode()

        l_sum_up = [(np.sum(i[0]), i[2]) for i in train_example]
        print(sum(i == (0, -1) or i == (-1, 1) for i in l_sum_up))    # second hand win
        print(sum(i == (0, 1) or i == (-1, -1) for i in l_sum_up))    # first hand win
