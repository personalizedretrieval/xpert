import numpy as np
import scipy.sparse as sp
import os
import torch
import tqdm
from collections import Counter
import time
from utils import (get_map_dict, get_reverse_map_dict, sample_batch_idx_skip_one, vmap_idx,
                    get_indices_map_embs, expand_csr_offsets, csr_to_pad_np, csr_select_one, )

from candidates import Candidates
from channels import ActivitySelector

def _apply_buffer(user_data, postimes, buffer=300):
    user_data.data = user_data.data * (user_data.data < (postimes[expand_csr_offsets(user_data.indptr)]-buffer))
    user_data.eliminate_zeros()


class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        # Path to read from first time
        self.user_dataset_file = config['user_dataset_file']
        # Path where pre-processed files are stored
        self.backup_dir = config['user_data_backup_dir']

        self.buffer = config['buffer']
        self.batch_on = config['batch_on'] if 'batch_on' in config else 'users'
        self.p_weight = config['p'] if 'p' in config else 1.0
        self.cluster_size = config['cluster_size'] if 'cluster_size' in config else -1
        self.multilabel = config['multilabel'] if 'multilabel' in config else False
        self.dedup_activity = False
        if 'dedup_activity' in config:
            self.dedup_activity = config['dedup_activity']
        print(f"Batching on {self.batch_on}")
        # Check if backup directory already consists the pre-processed files
        if self.load_backup():
            pass
        # If not load the raw data, process it and save in the backup directory
        else:
            raise NotImplementedError
        self.preprocess()

    def done_path(self):
        if self.multilabel:
            return 'done_multilabel.txt'
        else:
            return 'done.txt'

    # Load data from backup directory
    def load_backup(self):
        if not os.path.exists(os.path.join(self.backup_dir, self.done_path())):
            print('Backup not found')
            return False
        print('Backup found')
        start = time.time()
        self.user_ids = np.load(os.path.join(self.backup_dir, 'user_ids.npy'))
        if self.multilabel:
            self.posadids = sp.load_npz(os.path.join(self.backup_dir, 'posadids.npz'))
            self.posadids_unique = list(set(self.posadids.indices))
        else:
            self.posadids = np.load(os.path.join(self.backup_dir, 'posadids.npy'))
            self.posadids_unique = list(set(self.posadids))
        self.posadtimes = np.load(os.path.join(self.backup_dir, 'posadtimes.npy'))
        self.user_data = sp.load_npz(os.path.join(self.backup_dir, 'user_data.npz'))
        print(f'Time: {time.time() - start} user data loaded')
        return True

    # Pre-processing function
    def preprocess(self):
        if self.dedup_activity:
            dedup_user_data_file = os.path.join(self.backup_dir, 'dedup_user_data.npz')
            if os.path.exists(dedup_user_data_file):
                self.user_data = sp.load_npz(dedup_user_data_file)
                print("Loading dedup user data from backup")
            else:
                assert False, "error"
                rows, cols, data = [], [], []
                for i in tqdm.trange(self.user_data.shape[0]):
                    data_map = {}
                    for ind, dat in zip(self.user_data[i].indices, self.user_data[i].data):
                        if ind in data_map:
                            data_map[ind] = max(data_map[ind], dat)
                        else:
                            data_map[ind] = dat
                    rows.extend([i for _ in range(len(data_map.keys()))])
                    cols.extend(list(data_map.keys()))
                    data.extend(list(data_map.values()))
                self.user_data = sp.coo_matrix((data, (rows, cols)), shape=self.user_data.shape).tocsr()
                sp._sparsetools.csr_sort_indices(len(self.user_data.indptr) - 1, self.user_data.indptr,
                                                 self.user_data.data, self.user_data.indices)
                sp.save_npz(dedup_user_data_file, self.user_data)
        if self.buffer > 0:
            print(f"Applying buffer of {self.buffer} seconds to user data")
            self.user_data = _apply_buffer(self.user_data, self.posadtimes, self.buffer)
            sp._sparsetools.csr_sort_indices(len(self.user_data.indptr) - 1, self.user_data.indptr,
                                             self.user_data.data, self.user_data.indices)
        if not self.multilabel:
            self.posadids = sp.coo_matrix((np.ones_like(self.posadids),
                                           (np.arange(self.posadids.shape[0]),
                                            self.posadids))).tocsr()

    # Returns length of dataset
    def __len__(self):
        return self.user_data.shape[0]

    # torch dataset implements a __getitem__ function which is called repeatedly
    # batch_size number of time to create a batch.
    # Rest of processing and data being added to the batch is done in collator
    def __getitem__(self, idx):
        return self.user_ids[idx], self.user_data[idx], self.posadids[idx], self.posadtimes[idx]

class InMemoryCollator():
    '''
    Stitches a batch from the retrieved items
    '''
    def __init__(self, config, mode, activity_selector):
        self.mode = mode
        self.bsz = int(config['batch_size'])
        self.docs_features_file = config['docs_features_file']
        self.ad_features_file = config['ads_features_file']
        self.emb_dim = int(config['emb_dim'])
        self.collate_to_multiclass = config['collate_to_multiclass'] if 'collate_to_multiclass' in config else True
        self.num_negatives = int(config['num_negatives'])
        self.backup_dir = config['features_backup_dir']
        self.history_padding = config['history_padding']  # if >=0, returns history as padded np array
        self.batch_on = config['batch_on'] if 'batch_on' in config else 'users'
        self.negatives_type = config['negatives_type'] if 'negatives_type' in config else 'global'
        self.hard_neg_dist_thres = config['hard_neg_dist_thres'] if 'hard_neg_dist_thres' in config else 0.5
        self.activity_selector = activity_selector
        if self.num_negatives > 0:
            print("Num negatives: ", self.num_negatives)
            if (not self.collate_to_multiclass):
                raise NotImplementedError
        if self.load_backup():
            pass
        else:
            print("Num negatives here: ", self.num_negatives)
            raise NotImplementedError
        self.ad_map = get_map_dict(self.ad_ids)  # Dictionary: Ad_id -> Index of embedding
        self.reverse_ad_map = get_reverse_map_dict(self.ad_ids)  # Dictionary: Index -> Ad_id of training point
        self.doc_map = get_map_dict(self.doc_ids)  # Dictionary: Ad_id -> Index of embedding

    # Loads doc and ad - embeddings and ids, from the backup directory
    def load_backup(self):
        if not os.path.exists(os.path.join(self.backup_dir, 'done.txt')):
            print('Backup not found')
            return False
        print('Backup found')
        start = time.time()
        self.doc_ids = np.load(os.path.join(self.backup_dir, 'doc_ids.npy'))
        self.doc_emb = np.load(os.path.join(self.backup_dir, 'doc_emb.npy'))
        print(f'Time: {time.time() - start} doc features loaded')
        self.ad_ids = np.load(os.path.join(self.backup_dir, 'ad_ids.npy'))
        self.ad_emb = np.load(os.path.join(self.backup_dir, 'ad_emb.npy'))
        print(f'Time: {time.time() - start} ad features loaded')
        return True

    def __call__(self, batch_data):
        # Is called after get batch_size items are collected using __getitem__ of
        # getting history, novel behavoiours and doc features
        batch_size = len(batch_data)
        
        users = np.array([x[0] for x in batch_data])
        history = sp.vstack([x[1] for x in batch_data])
        positives = sp.vstack([x[2] for x in batch_data])

        if self.collate_to_multiclass:
            positives = csr_select_one(positives)

        history, novel, batch_doc_map, inverse_batch_doc_map, \
            batch_doc_embs, novel_importance, \
            selected_global_clusters = self.activity_selector({'history': history.copy(),
                                                               'doc_map': self.doc_map,
                                                               'doc_emb': self.doc_emb})

        # get positives and ad features
        # Array of ad_ids in the batch
        try:
            ad_indices = np.array(list(set(positives.indices)))
        except:
            ad_indices = np.array(list(set(positives)))
        # Returns updated ad_id->index map and embeddings that contains ads only in the batch
        batch_ad_map, batch_ad_embs = get_indices_map_embs(ad_indices, self.ad_map, self.ad_emb)

        # convert postives to indices of embeddings/training_points from original ids
        if isinstance(positives, sp.csr_matrix):
            positives.indices = vmap_idx(positives.indices, batch_ad_map).astype(np.int32)
            positives.data[positives.indices < 0] = 0
            positives.eliminate_zeros()
            pos_exists = np.array(positives.astype(np.bool).sum(1))[:, 0] > 0
        elif isinstance(positives, np.ndarray):
            positives = vmap_idx(positives, batch_ad_map).astype(np.int32)
            pos_exists = (positives >= 0)
        else:
            assert False, "unknown type in positives"

        # finding valid_points
        novel_exists = (np.array(novel.copy().astype(np.bool).sum(1))[:, 0] > 0)
        val_users = np.logical_and(novel_exists, pos_exists)

        history = history[val_users, :]
        novel = novel[val_users, :]
        if novel_importance is not None:
            novel_importance = novel_importance[val_users, :]
        positives = positives[val_users]
        users = users[val_users]
        batch_size = val_users.sum()

        if self.batch_on == 'users':
            # sampling negatives
            num_ads = batch_ad_embs.shape[0]
            num_docs = batch_doc_embs.shape[0]

            # not enough ads to sample negatives since negatives ads for each training point are sampled in-batch
            if num_ads <= max(self.num_negatives, 0):
                return None

            negatives = None
            if self.num_negatives > 0:
                if (not self.collate_to_multiclass):
                    raise NotImplementedError
                negatives = np.zeros((batch_size, self.num_negatives), dtype=np.int32)
                for i in range(batch_size):
                    negatives[i, :] = sample_batch_idx_skip_one(num_ads, self.num_negatives, positives[i])

        # merging ad and doc embeddings
        batch_embs = np.vstack([batch_doc_embs, batch_ad_embs])
        if isinstance(positives, sp.csr_matrix):
            positives.indices += num_docs
        elif isinstance(positives, np.ndarray):
            positives += num_docs
        else:
            assert False, "unknown type in positives"

        if self.num_negatives > 0:
            negatives += num_docs

        batch = {
            'batch_size': batch_size,
            'positives': positives,
            'negatives': negatives,
            'item_features': batch_embs,
            'novel_offsets': novel.indptr[:-1],
            'novel_items': novel.indices,
            'novel_importance': novel_importance,
            'selected_global_clusters': selected_global_clusters,
        }

        if self.history_padding > 0:
            history, history_lens = csr_to_pad_np(history, self.history_padding)
            batch.update({
                'history_lens': history_lens,
                'history_items': history,
            })
        else:
            batch.update({
                'history_offsets': history.indptr[:-1],
                'history_items': history.indices,
                'history_tfidfs': np.zeros_like(history.data, dtype=np.float32)+25.0,
            })

        if self.mode == 'eval':
            batch.update({
                'ad_map': batch_ad_map,
                # 'doc_map' : batch_doc_map,  # including this slows down the dataloader
                'num_docs': num_docs,
                'user_ids': users
            })
        return batch



class InMemoryDataLoader():
    '''
    Inputs:
        config - Configuration
        mode - Evaluation (will contain candidates additionally for retrieval) or Training
    '''
    def __init__(self, config, mode, activity_selector, candidates_config=None):
        self.mode = mode
        self.config = config
        # Used to rerieve a single item
        self.dataset = InMemoryDataset(config)
        # Used to stitch the batch and perform operations on a batch before returning
        self.col = InMemoryCollator(config, mode, activity_selector)
        self.bsz = int(config['batch_size'])
        self.num_workers = int(config['num_workers'])
        self.shuffle = False
        if int(config['shuffle']) == 1:
            self.shuffle = True
        # Pin memory is for optmization by pinning memory on non-paged CPU memory to avoid GPU<->CPU cycle
        self.pin_mem = False
        if int(config['pin_mem']) == 1:
            self.pin_mem = True

        self.dl = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bsz,
            num_workers=self.num_workers,
            collate_fn=self.col,
            shuffle=self.shuffle,
            pin_memory=self.pin_mem
        )

    # This is called every epoch, which returns an iterator
    # Internally, next is called on this iterator to build and return batches
    def __iter__(self):
        return iter(self.dl)

    def __len__(self):
        return len(self.dl)

    def join(self):
        pass

    def feat_type(self):
        return self.config['features_type']
    

class DatasetMain():
    '''
    General dataset class which will be iterable and would also allow NNS
    over candidate set as an option
        config: config setting all the parameters needed for creating the dataset
        candidates: boolean. If true will load candidates and creates a NNS structure
                    Candidates are used during evaluation, where these are the candidates for retrieval.
    '''
    def __init__(self, config, candidates=False):
        self.candidates = candidates
        self.activity_selector = ActivitySelector(config['activity_selector'])
        config_data = config['data']
        self.dataset = InMemoryDataLoader(config_data,
                                          mode=('eval' if candidates else 'train'),
                                          activity_selector=self.activity_selector,
                                          candidates_config=config['candidates'])

        if candidates:
            self.candidates = Candidates(config['candidates'])

    # Returns iterator - Called every epoch
    def __iter__(self):
        self.iter_dataset = iter(self.dataset)
        return self

    # Iterates over the iterator - Called every iteration
    # Returns 1 batch at a time
    def __next__(self):
        batch = next(self.iter_dataset)
        while batch is None:
            batch = next(self.iter_dataset)
        if self.candidates:
            batch['gt_smat'] = self.candidates.map_pos(batch)
        return batch

    def getnns(self, q, novel_offsets=None, k=10, sampling_probabilities=None, analysis=False):
        return self.candidates.getnns(q, novel_offsets, k, sampling_probabilities, analysis)

    def join(self):
        self.dataset.join()