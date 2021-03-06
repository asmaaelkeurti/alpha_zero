import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import tensorflow as tf
from utils import *
from NeuralNet import NeuralNet
from keras.callbacks import EarlyStopping
import argparse
from GobangNNet import GobangNNet as onnet
from keras.models import load_model
sys.path.append('..')

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 200,
    'batch_size': 512,
    'cuda': True,
    # 'num_channels': 512,
    'num_channels': 20
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.graph = tf.get_default_graph()
        self.nnet = onnet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards, y=[target_pis, target_vs], validation_split=0.2, batch_size=args.batch_size,
                            epochs=args.epochs, callbacks=[EarlyStopping(patience=2)])
        # callbacks=[EarlyStopping()]

    def evaluate_model(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        return self.nnet.model.evaluate(x=input_boards, y=[target_pis, target_vs])

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]
        with self.graph.as_default():
            # run
            # self.nnet.model._make_predict_function()
            pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)

    def save_model(self, folder='checkpoint', filename='model'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save(filepath)

    def load_model(self, folder='checkpoint', filename='model'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model = load_model(filepath)

