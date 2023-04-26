# Candidates
import os
import numpy as np
import scipy.sparse as sp
import re

from utils import HNSW
from collections import Counter
from tqdm.notebook import tqdm

def inv_dict(_dict):
    return {v: k for k, v in _dict.items()}

def _read_embeddings(fin, emb_dim):
    
    ids = []
    emb = []

    for line in tqdm(fin):
        _split = line.split('\t', -1)
        if(len(_split) != 2):
            continue
        embs = _split[1].split()
        if(len(embs) != emb_dim):
            continue
            
        ids.append(int(_split[0]))
        
        for i in range(0, emb_dim):
            embs[i] = float(embs[i])
        
        emb.append(embs)
        
    return np.array(ids, dtype=np.int32), np.array(emb, dtype=np.float32).reshape((-1, emb_dim))

def read_embs_txt(fname, emb_dim=64):
    with open(fname, 'r') as fin:
        ids, emb = _read_embeddings(fin, emb_dim)
    return ids, emb

def read_embs(fname, emb_dim=64):
    if fname.endswith('txt') or fname.endswith('tsv'):
        ids, embs = read_embs_txt(fname, emb_dim)
    elif fname.endswith('npy'):
        embs = np.load(fname)
        ids = np.arange(embs.shape[0])
    else:
        raise NotImplementedError
    return ids, embs

class NNS():
    def __init__(self, config, embs):
        self.config = config
        self.retrieval_stategy = config['retrieval_stategy']
        self.max_retrievals_per_user = config['max_retrievals_per_user'] \
            if "max_retrievals_per_user" in config else None
        self.less_nn_retrievals = 0
        if config['type'] == 'hnsw':
            self.hnsw = HNSW(M=config['M'], efC=config['efC'], efS=config['efS'], num_threads=config['num_threads'])
            if os.path.exists(config['hnsw_backup']):
                self.hnsw.load(config['hnsw_backup'])
            else:
                raise NotImplementedError
        self.num_candidates = embs.shape[0]

    def getnns(self, q, novel_offsets=None, k=10, sampling_probabilities=None, analysis=False):
        if re.compile(r'importance_raise\d.(\d)+_sampling').search(self.retrieval_stategy):
            imp_power = float(re.compile(r'\d.(\d)+').search(self.retrieval_stategy).group())
            sampling_probabilities.data = sampling_probabilities.data**imp_power
            return self.getnns_helper(q=q,
                                      novel_offsets=novel_offsets,
                                      k=k,
                                      sampling_probabilities=sampling_probabilities,
                                      analysis=analysis)
        else:
            raise NotImplementedError

    def getnns_from_data(self, q, k):
        if self.config['type'] == 'hnsw':
            return self.hnsw.predict(q, k)
        else:
            raise NotImplementedError

    def getnns_helper(self, q, novel_offsets=None, k=10, sampling_probabilities=None, analysis=False):
        '''
            Sampling_probailities: Sampling weights for each embedding.
                                None means uniform sampling
                                Else, an csr array of the same size as novel is expected
                                specifying weights of each embedding

        '''
        if sampling_probabilities is not None:
            # Converting sampling probabilities to the number of retrievals for each embedding
            all_samples = np.array([], dtype=np.int32)
            indptr = np.array([0])
            for i in range(sampling_probabilities.shape[0]):
                # Use all slots
                if self.max_retrievals_per_user:
                    retrieval_per_user = self.max_retrievals_per_user
                else:
                    retrieval_per_user = (sampling_probabilities.indptr[i+1]-sampling_probabilities.indptr[i])*k
                # Max k slots per event
                # retrieval_per_user = (sampling_probabilities.indptr[i+1]-sampling_probabilities.indptr[i])*k
                # if max_retrievals_per_user:
                #     retrieval_per_user = min(retrieval_per_user, max_retrievals_per_user)

                w = np.array(sampling_probabilities.data[sampling_probabilities.indptr[i]:
                                                         sampling_probabilities.indptr[i+1]])
                w = (w/np.sum(w))
                samples = (w*retrieval_per_user).astype(int)
                samples_dict = Counter(np.random.choice(range(0, len(w), 1),
                                       int(retrieval_per_user-np.sum(samples)), True, w))
                samples += np.array([samples_dict[i] if i in samples_dict.keys()
                                    else 0 for i in range(0, len(w), 1)],
                                    dtype=np.int32)
                all_samples = np.append(all_samples, samples)
                indptr = np.append(indptr, indptr[-1]+retrieval_per_user)
            assert q.shape[0] == all_samples.shape[0], "Incorrect length of samples"

            indices, distances = self.getnns_from_data(q, int(max(all_samples)))  # #embeddings(q)*max(ind_samples)
            for i in range(0, indices.shape[0]):
                if np.any(np.where(indices[i] == 0)[0] < all_samples[i]):
                    self.less_nn_retrievals += 1
            indices = np.hstack([indices[i][0:all_samples[i]] for i in range(0, indices.shape[0])])
            distances = np.hstack([distances[i][0:all_samples[i]] for i in range(0, distances.shape[0])])

            distances = 1-distances
            return sp.csr_matrix((distances, indices, indptr),
                                 shape=(len(novel_offsets), self.num_candidates)), self.less_nn_retrievals
        else:
            raise NotImplementedError


class Candidates():
    def __init__(self, config):
        self.config = config
        self.ids, self.embs = read_embs(config['candidates_embs_file'], config.get('emb_dim', None))
        
        self.candidate_map = {x: i for i, x in enumerate(self.ids)}
        self.num_candidates = self.embs.shape[0]
        self.nns = NNS(config['nns'], self.embs)

    def getnns(self, q, novel_offsets, k=10, sampling_probabilities=None, analysis=False):
        return self.nns.getnns(q, novel_offsets, k, sampling_probabilities, analysis)

    def map_pos(self, batch):
        if isinstance(batch['positives'], np.ndarray):
            data = np.ones(batch['batch_size'])
            inv_map = inv_dict(batch['ad_map'])
            positive_ids = batch['positives']-batch['num_docs']
            indices = np.array([self.candidate_map[inv_map[x]] for x in positive_ids])
            indptr = np.arange(0, batch['batch_size']+1)
            return sp.csr_matrix((data, indices, indptr),
                                 shape=(batch['batch_size'], self.num_candidates))
        else:
            new_pos = batch['positives'].copy()
            inv_map = inv_dict(batch['ad_map'])
            new_pos.indices -= batch['num_docs']
            new_pos.indices = np.array([self.candidate_map[inv_map[x]]
                                        for x in new_pos.indices])
            new_pos._shape = (batch['batch_size'], self.num_candidates)
            return new_pos