import heapq
import pdb

import Evaluate
from Dataset import Dataset
from DeepICF_a import parse_args, DeepICF_a
import tensorflow as tf
import numpy as np
from functools import partial
from itertools import combinations
from sklearn.preprocessing import normalize


def jaccard_score(a, b):
    set_a, set_b = set(a), set(b)
    return len(set_a & set_b) / len(set_a | set_b)


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
    plt.ylabel("Average Jaccard index for top 5 attention weights")
    plt.show()


def avg_variation(all_vals):
    variations = []
    for comb in list(combinations(range(len(all_vals)), 2)):
        variations.append(np.abs(all_vals[comb[0]] - all_vals[comb[1]]))
    return np.mean(variations)


def plot_avg_variation(all_models_predictions):
    all_avg_preds = []
    for column in all_models_predictions.T:
        mean = np.mean(column)
        # std = np.mean(np.abs(column - mean))
        # std = np.std(column)
        std = avg_variation(column)
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
    # num_models = len(models_paths)
    num_models = 10
    top_n_attention_weights = 5
    bound_top_n_indices = partial(top_n_indices, top_n_attention_weights)
    all_predictions = []
    all_attention_vectors = []
    for model_path in models_paths[:num_models]:
        print('Getting predictions for ', model_path)
        preds = get_prediction_scores(model, dataset, model_path)
        all_predictions.append(preds)

        attention_vectors = get_attention_score_for_tests(model, dataset,
                                                          model_path)
        all_attention_vectors.append(attention_vectors)
    all_predictions = np.array(all_predictions).reshape(num_models, len(dataset.testRatings)).T
    combins = list(combinations(range(num_models), 2))
    conf_vs_jaccard_map = {(round(k, 1)): [0, 0] for k in np.arange(0, 1, 0.1)}
    for test_num in range(len(dataset.testRatings)):
        jaccard_scores = []
        avg_prediction_score = np.mean(all_predictions[test_num])
        avg_prediction_score_round = avg_prediction_score // 0.1 / 10
        for comb in combins:
            model_i, model_j = comb[0], comb[1]
            atn_vec_i, atn_vec_j = all_attention_vectors[model_i][test_num], all_attention_vectors[model_j][test_num]
            atn_vec_i_norm, atn_vec_j_norm = np.squeeze(normalize(atn_vec_i[None, :])), np.squeeze(
                normalize(atn_vec_j[None, :]))
            top_n_i, top_n_j = bound_top_n_indices(atn_vec_i_norm), bound_top_n_indices(atn_vec_j_norm)
            jaccard_scores.append(jaccard_score(top_n_i, top_n_j))
        avg_jaccard_score = np.mean(jaccard_scores)
        conf_vs_jaccard_map[avg_prediction_score_round][
            0] += avg_jaccard_score
        conf_vs_jaccard_map[avg_prediction_score_round][1] += 1
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
    preds = get_prediction_scores(model, dataset, eg_model_path)
    conf_vs_max_attentions_sum = {(round(k, 1)): [0, 0] for k in np.arange(0, 1, 0.1)}
    for i in range(len(attention_vectors)):
        vector = attention_vectors[i]
        # n_l = int(0.05 * len(vector) + 1)
        n_l = 5
        normalized_vec = np.squeeze(normalize(vector[None, :], 'l1'))
        print(np.sum(normalized_vec))
        n_largest = np.sum(heapq.nlargest(n_l, normalized_vec))
        pred = preds[i] // 0.1 / 10
        conf_vs_max_attentions_sum[pred][0] += n_largest
        conf_vs_max_attentions_sum[pred][1] += 1
    plot_map(conf_vs_max_attentions_sum)
