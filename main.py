from Coach import Coach
from GobangGame import GobangGame as Game
from NNet import NNetWrapper as nn
from AutoRun import AutoRun
from multiprocessing import Pool


args = {
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 30,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 1000,
    'arenaCompare': 64,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
}

if __name__ == "x":
    g = Game(8)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.load_train_examples()
    c.learn()

if __name__ == '__main__':
    g = Game(8)
    auto_run = AutoRun(g, args)

    # print(auto_run.arena_process(12, 'manual_trained_model', 'model_auto_6', verbose=True))

    result = auto_run.arena_process_parallel(12, 'manual_trained_model', 'model_auto_6')
    print(result)

    # auto_run.generate_data_parallel(400, 'model_auto_6', 'train_examples_auto_6')



