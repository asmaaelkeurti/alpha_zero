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


if __name__ == '__main__':
    trainExamples = merge_data(["./temp/train_examples_auto_6"])
    shuffle(trainExamples)
    print(len(trainExamples))

    g = Game(8)
    nnet = nn(g)
    nnet.train(trainExamples)
    nnet.save_model(filename="manual_trained_model")


if __name__ == 'x':
    trainExamples = merge_data(merge_data(["./temp/train_examples_auto_6"]))
    g = Game(8)
    nnet = nn(g)
    nnet.load_model(filename="manual_trained_model")
    print(nnet.evaluate_model(trainExamples))

if __name__ == 'x':
    examples = merge_data(["./temp/train_examples_auto_6"])
    l_sumup = [(np.sum(i[0]), i[2]) for i in examples]
    print(sum(i == (0, -1) or i == (-1, 1) for i in l_sumup))
    print(sum(i == (0, 1) or i == (-1, -1) for i in l_sumup))

