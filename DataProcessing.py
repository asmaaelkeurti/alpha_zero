import pickle
from NNet import NNetWrapper as nn
from GobangGame import GobangGame as Game


def merge_data():
    file_list = ["./temp/train_examples_2", "./temp/train_examples_2_1", "./temp/train_examples_3"]
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
    trainExamples = merge_data()

    print(len(trainExamples))

    g = Game(8)
    nnet = nn(g)

    nnet.train(trainExamples)

    nnet.save_model(filename="second_model")
