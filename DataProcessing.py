import pickle
from NNet import NNetWrapper as nn
from GobangGame import GobangGame as Game
from random import shuffle
import numpy as np


def merge_data(file_list):
    objects = []
    trainExamples = []

    for f in file_list:
        with(open(f, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break

    for o in objects:
        trainExamples += o

    return trainExamples


def get_balanced_data(trainExamples):
    trainExamples = [[x[0], x[1], x[2], np.sum(x[0])] for x in trainExamples]

    trainExamples_left = []
    trainExamples_right = []

    for e in trainExamples:
        if (e[3], e[2]) == (0, -1) or (e[3], e[2]) == (-1, 1):
            trainExamples_left.append(e[:-1])
        else:
            trainExamples_right.append(e[:-1])

    length = min(len(trainExamples_left), len(trainExamples_right))
    shuffle(trainExamples_left)
    shuffle(trainExamples_right)

    return trainExamples_left[:length] + trainExamples_right[:length]


if __name__ == '__main__':
    trainExamples = merge_data(["./temp/train_examples_auto_1"])
    shuffle(trainExamples)
    print(len(trainExamples))

    g = Game(15)
    nnet = nn(g)
    nnet.train(trainExamples)
    nnet.save_model(filename="manual_trained_model")


if __name__ == 'x':
    trainExamples = merge_data(["./temp/train_examples_auto_10"])
    g = Game(15)
    nnet = nn(g)
    nnet.load_model(filename="manual_trained_model")
    print(nnet.evaluate_model(trainExamples))

if __name__ == 'x':
    examples = merge_data(["./temp/train_examples_auto_10"])
    l_sumup = [(np.sum(i[0]), i[2]) for i in examples]
    print(sum(i == (0, -1) or i == (-1, 1) for i in l_sumup))
    print(sum(i == (0, 1) or i == (-1, -1) for i in l_sumup))

