import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle
import codecs

users_attr_to_index = {
  "Gender": 1,
  "Age": 2,
  "Occupation": 3,
  "Zip": 4
}


def getMovieIds(movie_types=None):
  if movie_types is None:
    movie_types = []
  #   Mapping is from NCF ids to actual movielens ids
  with open('Data/movie_mapping.json') as f:
    movie_mapping_dict = json.load(f)
  if len(movie_types) == 0:
    return list(map(int, movie_mapping_dict.keys()))

  inv_map = {v: k for k, v in movie_mapping_dict.items()}

  movie_ids = []

  with codecs.open("Data/movies.dat", "r", encoding='utf-16',
                   errors='ignore') as f:
    line = f.readline()

    while line:
      arr = line.split("::")
      cur_movie_type = arr[2]
      ml_id = int(arr[0])
      movie_type_matches = True
      for movie_type in movie_types:
        if movie_type not in cur_movie_type:
          movie_type_matches = False
          break
      if movie_type_matches and (ml_id in inv_map):
        movie_ids.append(int(inv_map[ml_id]))

      line = f.readline()

  return movie_ids


def perfTSNE(embedding_vectors, n_components=2):
  return TSNE(n_components=n_components, learning_rate=300.0,
              n_iter=500).fit_transform(
    embedding_vectors)


def perfPCA(embedding_vectors):
  return PCA(n_components=2).fit_transform(embedding_vectors)


def plotPoints(X_embedded, color, label):
  if len(X_embedded) != 0:
    plt.plot(X_embedded[:, 0], X_embedded[:, 1], color, label=label)


def getIdsEmbeddings(X_embedded, ids):
  ret = []
  for id in ids:
    ret.append(X_embedded[id])
  return np.array(ret)


def pickle_data_to_file(data, file_name):
  with open(file_name, 'wb') as file:
    pickle.dump(data, file)


def get_movielens_id_to_movie_map():
  movielens_id_to_movie = dict()
  import codecs

  with codecs.open("Data/movies.dat", "r", encoding='utf-16',
                   errors='ignore') as f:
    line = f.readline()

    while line:
      arr = line.split("::")
      ml_id = int(arr[0])
      movielens_id_to_movie[ml_id] = arr

      line = f.readline()
  return movielens_id_to_movie


def get_ncf_id_to_movie_map():
  #   Mapping is from NCF ids to actual movielens ids
  with open('Data/movie_mapping.json') as f:
    ncf_ml_id_map = json.load(f)

  ml_id_to_movie = get_movielens_id_to_movie_map()
  ncf_id_to_movie = dict()
  for ncf_id, ml_id in ncf_ml_id_map.items():
    ncf_id = int(ncf_id)
    ncf_id_to_movie[ncf_id] = ml_id_to_movie[ml_id]
  return ncf_id_to_movie


def get_id_to_user_map():
  id_to_user_map = dict()
  with codecs.open("Data/users.dat", "r", encoding='utf-8',
                   errors='ignore') as f:
    line = f.readline()
    while line:
      arr = line.split("::")
      id = int(arr[0]) - 1
      id_to_user_map[id] = arr
      line = f.readline()
  return id_to_user_map


def attr_cond_satisfied(attr, arr, cond):
  val = arr[users_attr_to_index[attr]]
  if attr == "Age":
    return cond[0] <= int(val) <= cond[1]
  return cond == val

# Example types = {
#                 "Age":[18,25],
#                 "Gender": "M
#                  }
def get_users(types=None):
  if types is None:
    types = dict()
  users = []
  with codecs.open("Data/users.dat", "r", encoding='utf-8',
                   errors='ignore') as f:
    line = f.readline()
    while line:
      add_to_list = True
      arr = line.split("::")
      for attr, cond in types.items():
        if not attr_cond_satisfied(attr, arr, cond):
          add_to_list = False
          break

      if add_to_list:
        users.append(int(arr[0]) - 1)
      line = f.readline()
  return users


def load_emb(type='item'):
  with open('Embeddings/%s_embs_d_a.pkl' % type, 'rb') as pkl_file:
    embs = pickle.load(pkl_file)
    return embs