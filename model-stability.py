import Evaluate
from Dataset import Dataset
from DeepICF_a import parse_args, DeepICF_a
import tensorflow as tf
import numpy as np
from functools import partial
from itertools import combinations
from sklearn.metrics import jaccard_similarity_score


def ignore_vals(x):
    return np.exp(x)


def softmax(x):
    e_x = np.array(list(map(ignore_vals, x - np.max(x))))
    return e_x / e_x.sum(axis=0)


def top_n_indices(n, arr):
    return np.argpartition(arr, -n)[-n:]


def get_models(filename):
    model_files = []
    with open(filename, "r") as f:
        line = f.readline()
        while line is not None and line != "":
            if "Model saved as" in line:
                words = line.split(" ")
                model_files.append(words[3][:-1])

            line = f.readline()
    return model_files


def get_feed_dict(_model, test_rating_index, dataset):
    rating = dataset.testRatings[test_rating_index]
    user = dataset.trainList[test_rating_index]
    num_idx_ = len(user)
    gtItem = rating[1]
    items = [gtItem, gtItem]
    # Get prediction scores
    num_idx = np.full(len(items), num_idx_, dtype=np.int32)[:, None]
    user_input = []
    for i in range(len(items)):
        user_input.append(user)
    user_input = np.array(user_input)
    item_input = np.array(items)[:, None]
    feed_dict = {_model.user_input: user_input, _model.num_idx: num_idx,
                 _model.item_input: item_input, _model.is_train_phase: False}
    return feed_dict


def get_all_users_items_feed_dict(_model, num_items):
    user_input = np.array([np.arange(0, num_items)] * num_items)
    item_input = np.arange(0, num_items)[:, None]
    num_idx = np.full(num_items, num_items, dtype=np.int32)[:, None]
    feed_dict = {_model.user_input: user_input, _model.num_idx: num_idx,
                 _model.item_input: item_input, _model.is_train_phase: False}
    return feed_dict


def plot_map(conf_vs_acc_map):
    import matplotlib.pyplot as plt

    conf_map = {k: v[0] / (v[1] + 0.00001)
                for k, v in conf_vs_acc_map.items()}
    plt.bar(range(len(conf_map.keys())), list(conf_map.values()),
            tick_label=list(conf_map.keys()))
    plt.xlabel("Positive label prediction score")
    plt.ylabel("Average Jaccard similarity score")
    plt.show()


def plot_avg_variation(all_models_predictions):
    all_avg_preds = []
    for column in all_models_predictions.T:
        mean = np.mean(column)
        std = np.mean(np.abs(column - mean))
        # std = np.std(column)
        all_avg_preds.append((mean, std))
    conf_vs_acc_map = {(round(k, 1)): [0, 0] for k in np.arange(0, 1, 0.1)}

    for avg_pred in all_avg_preds:
        conf_vs_acc_map[avg_pred[0] // 0.1 / 10][0] += avg_pred[1]
        conf_vs_acc_map[avg_pred[0] // 0.1 / 10][1] += 1

    plot_map(conf_vs_acc_map)


def get_prediction_scores(model, dataset, weight_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        all_preds = []
        for test_idx in range(len(dataset.testRatings)):
            output = sess.run([model.output],
                              feed_dict=get_feed_dict(model, test_idx,
                                                      dataset))[0]
            all_preds.append(output[-1][0])
        return all_preds


def get_attention_scores(model, dataset, weight_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        my_feed_dict = get_all_users_items_feed_dict(model, dataset.num_items)
        preA, maskMat = sess.run([model.preA, model.mask_mat],
                                 feed_dict=my_feed_dict)
        print(preA)


def get_attention_score_for_tests(model, dataset, weight_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        attention_vectors = []
        for test_idx in range(len(dataset.testRatings)):
            attention_vector = np.squeeze(sess.run([model.A],
                                                   feed_dict=get_feed_dict(
                                                       model, test_idx,
                                                       dataset))[0])[-1]
            attention_vectors.append(attention_vector)
        return attention_vectors


def get_pre_attention_score_for_tests(model, dataset, weight_path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, weight_path)
        attention_vectors = []
        for test_idx in range(len(dataset.testRatings)):
            attention_vector = np.squeeze(sess.run([model.preA],
                                                   feed_dict=get_feed_dict(
                                                       model, test_idx,
                                                       dataset))[0])[-1]
            attention_vectors.append(attention_vector)
        return attention_vectors


def plot_jaccard_vs_pred_score():
    num_models = len(models_paths)
    top_n_attention_weights = 1
    bound_top_n_indices = partial(top_n_indices, top_n_attention_weights)
    all_max_indices = []
    all_predictions = []
    for model_path in models_paths[:num_models]:
        print('Getting predictions for ', model_path)
        preds = get_prediction_scores(model, dataset, model_path)
        all_predictions.append(preds)

        attention_vectors = get_attention_score_for_tests(model, dataset,
                                                          model_path)
        max_indices = list(
            map(bound_top_n_indices,
                attention_vectors))  # shape (num_tests,top_n_weights)
        all_max_indices.append(max_indices)
    all_max_indices = np.array(all_max_indices).reshape(num_models, 6040,
                                                        top_n_attention_weights)
    all_predictions = np.array(all_predictions).reshape(num_models, 6040).T

    combins = list(combinations(range(num_models), 2))

    conf_vs_jaccard_map = {(round(k, 1)): [0, 0] for k in np.arange(0, 1, 0.1)}
    for test_num in range(all_max_indices.shape[1]):
        total_jaccard_score = 0.0
        for comb in combins:
            total_jaccard_score += jaccard_similarity_score(
                all_max_indices[comb[0]][test_num],
                all_max_indices[comb[1]][test_num])
        avg_jaccard_score = total_jaccard_score / len(combins)
        avg_prediction_score = np.mean(all_predictions[test_num])
        conf_vs_jaccard_map[avg_prediction_score // 0.1 / 10][
            0] += avg_jaccard_score
        conf_vs_jaccard_map[avg_prediction_score // 0.1 / 10][1] += 1
    plot_map(conf_vs_jaccard_map)


if __name__ == '__main__':
    models_paths = get_models(
        './seeded-trainlogs-deepICF-a-2opt-cross-loss.txt')
    args = parse_args()

    dataset = Dataset(args.path + args.dataset)
    model = DeepICF_a(dataset.num_items, args)
    model.build_graph()
    eg_model_path = models_paths[1]
    attention_vectors = get_attention_score_for_tests(model, dataset,
                                                      eg_model_path)
    count = 0
    for vector in attention_vectors:
        normalized_vec = softmax(vector)
        if np.max(normalized_vec) > 0.99:
            count += 1
    print(count)
