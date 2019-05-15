from Dataset import Dataset
from DeepICF_a import parse_args, DeepICF_a, training
import os
import tensorflow as tf
import numpy as np
import random

if __name__ == '__main__':

    args = parse_args()
    regs = eval(args.regs)
    seeds = eval(args.seeds)
    print("Seeds", seeds)
    for iteration_num in range(len(seeds)):
        tf.reset_default_graph()
        os.environ['PYTHONHASHSEED'] = str(seeds[iteration_num])
        tf.set_random_seed(seeds[iteration_num])
        np.random.seed(seeds[iteration_num])
        random.seed(seeds[iteration_num])
        print("Iteration:", iteration_num)
        dataset = Dataset(args.path + args.dataset)
        model = DeepICF_a(dataset.num_items, args)
        model.build_graph()
        best_hr, best_ndcg = training(0, model, dataset, args.epochs,
                                      args.num_neg)

        print("End. best HR = %.4f, best NDCG = %.4f" % (best_hr, best_ndcg))
