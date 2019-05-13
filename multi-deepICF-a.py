from Dataset import Dataset
from DeepICF_a import parse_args, DeepICF_a, training

if __name__ == '__main__':

    args = parse_args()
    regs = eval(args.regs)
    for iteration_num in range(10):
        print("Iteration:", iteration_num)
        dataset = Dataset(args.path + args.dataset)
        model = DeepICF_a(dataset.num_items, args)
        model.build_graph()
        best_hr, best_ndcg = training(0, model, dataset, args.epochs,
                                      args.num_neg)

        print("End. best HR = %.4f, best NDCG = %.4f" % (best_hr, best_ndcg))
