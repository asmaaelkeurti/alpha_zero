from Coach import Coach
from GobangGame import GobangGame as Game
from NNet import NNetWrapper as nn
from AutoRun import AutoRun
from multiprocessing import Pool


args = {
    'numIters': 10,
    'numEps': 10,
    'tempThreshold': 25,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 300,
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

    # arena_result = auto_run.arena_process_parallel([12, 'manual_trained_model', 'model_auto_6', True])

    with Pool(8) as p:
        arena_result = p.map(auto_run.arena_process_parallel,
            [[12, 'manual_trained_model', 'model_auto_5', False] for _ in range(8)])
    print(arena_result)
