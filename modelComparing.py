from NNet import NNetWrapper as nn
from GobangGame import GobangGame as Game
from Arena import Arena
import numpy as np
from MCTS import MCTS
from GobangGame import display
from multiprocessing import Process, Pool, Lock
from Coach import Coach
import os
import pickle


args = {
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 30,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 300,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

}


def arena_process(i):
    g = Game(8)

    nnet = nn(g)
    nnet.load_model(filename=("model_auto_" + str(i+1)))
    nmcts = MCTS(g, nnet, args)

    pnet = nn(g)
    if i != 0:
        pnet.load_model(filename=("model_auto_" + str(i)))
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


def generate_data(l, model_iter):
    g = Game(8)
    nnet = nn(g)
    nnet.load_model(filename=("model_auto_" + str(model_iter + 1)))

    c = Coach(g, nnet, args)
    train_example = c.execute_episode()

    l.acquire()
    try:
        folder = args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder + ("train_examples_auto_" + str(model_iter + 1)))
        with open(filename, "ab+") as f:
            pickle.dump(train_example, f)
    finally:
        l.release()


def merge_data(model_iter):
    file_list = ["./temp/train_examples_auto_" + str(model_iter+1)]
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
    better_model = False

    for i in range(3, 100):
        with Pool(8) as p:
            result = p.map(arena_process, [i for _ in range(8)])

        win_1 = sum([i[0] for i in result])
        win_2 = sum([i[1] for i in result])

        print('model', i+1, 'win_1', win_1, 'win_2', win_2)

        if win_1 < win_2*0.9:
            lock = Lock()

            for iteration in range(200):
                jobs = []

                for _ in range(8):
                    p = Process(target=generate_data, args=(lock, i))
                    jobs.append(p)
                    p.start()

                for job in jobs:
                    job.join()

            trainExamples = merge_data(i)
            print(len(trainExamples))
            g = Game(8)
            nnet = nn(g)
            nnet.train(trainExamples)
            nnet.save_model(filename="model_auto_" + str(i+2))
        else:
            break

        print(i, 'one model')


if __name__ == 'x':
    with Pool(8) as p:
        result = p.map(arena_process, range(8))

        win_1 = sum([i[0] for i in result])
        win_2 = sum([i[1] for i in result])

        print(win_1)
        print(win_2)
