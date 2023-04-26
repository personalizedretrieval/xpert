import numpy as np
import nmslib
import functools
import operator
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from envyaml import EnvYAML

# Returns key->index dictionary
def get_map_dict(ids):
    return {idx: i for i, idx in enumerate(ids)}

# Returns index->key dictionary
def get_reverse_map_dict(ids):
    return {i: idx for i, idx in enumerate(ids)}

# Returns indices from a list of key and key->indices dictionary
def map_idx(idx, map_dict):
    return map_dict.get(idx, -1)

# sample num_sample indices from 0..batch_size ensuring skip_index is not sampled
def sample_batch_idx_skip_one(batch_size, num_sample, skip_index):
    # random indices ensuring there is no batch_size-1
    sampled_indices = np.random.choice(batch_size-1, num_sample, replace=False)
    # rotate batch_size-1 to skip_index
    return (sampled_indices+1+skip_index) % batch_size

def expand_csr_offsets(offsets):
    expanded = np.empty(offsets[-1], dtype=int)
    sp._sparsetools.expandptr(len(offsets)-1, offsets, expanded)
    return expanded

def csr_to_pad_np(csr, num_pad):
    arr = np.zeros((csr.shape[0], num_pad), dtype=np.int32)-1
    lens = np.zeros(csr.shape[0], dtype=np.int32)
    for i in range(csr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i+1]
        start = max(start, end-num_pad)
        arr[i][:end-start] = csr.indices[start:end]
        lens[i] = end-start
    return arr, lens

def csr_select_one(csr):
    arr = np.zeros(csr.shape[0], dtype=np.int32)
    for i in range(csr.shape[0]):
        start, end = csr.indptr[i], csr.indptr[i+1]
        arr[i] = np.random.choice(csr.indices[start:end])
    return arr
    
vmap_idx = np.vectorize(map_idx)

# Trim id_map and embeddings based on indices (list of keys)
# indice - List of ids/keys
# id _map - Dictionary: id/key -> index of training point/embedding
# embeddings - embeddings of docs/ads
def get_indices_map_embs(indices, id_map, embeddings, return_indices=False):
    if id_map is not None:
        emb_indices = vmap_idx(indices, id_map)
        valid_idx = (emb_indices >= 0)
        indices, emb_indices = indices[valid_idx], emb_indices[valid_idx]
    else:
        emb_indices = indices
    indices_embs = embeddings[emb_indices]
    indices_map = get_map_dict(indices)
    if not return_indices:
        return indices_map, indices_embs
    else:
        return indices_map, indices_embs, indices


class HNSW(object):
    def __init__(self, M, efC, efS, num_threads):
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS

    def fit(self, data, print_progress=True):
        self.index.addDataPointBatch(data)
        self.index.createIndex(
            {'M': self.M,
             'indexThreadQty': self.num_threads,
             'efConstruction': self.efC},
            print_progress=print_progress
            )

    def _filter(self, output, k):
        indices = np.zeros((len(output), k), dtype=np.int32)
        distances = np.zeros((len(output), k), dtype=np.float32)
        for idx, item in enumerate(output):
            # Use padding instead of zeros for empty slots
            if item[0].shape[0] < k:
                indices[idx][0:item[0].shape[0]] = item[0]
                distances[idx][0:item[1].shape[0]] = item[1]
            else:
                indices[idx] = item[0]
                distances[idx] = item[1]
        return indices, distances

    def predict(self, data, k=None):
        self.index.setQueryTimeParams({'efSearch': self.efS})
        if k is None:
            k_to_use = self.efS
        else:
            k_to_use = k
        output = self.index.knnQueryBatch(
            data, k=k_to_use, num_threads=self.num_threads
            )
        indices, distances = self._filter(output, k=k_to_use)
        return indices, distances

    def save(self, fname):
        nmslib.saveIndex(self.index, fname)

    def load(self, fname):
        nmslib.loadIndex(self.index, fname)
