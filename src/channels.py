import numpy as np
import scipy.spatial as spt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize
import math
import scipy.sparse as sp
from utils import vmap_idx, get_map_dict, get_indices_map_embs
from collections import deque, OrderedDict

class Channels():
    def __init__(self, config):
        self.initial_novel = config['initial_novel']
        self.cluster_mode = config['mode']
        self.clustering_algo = config['clustering_algo']
        if self.cluster_mode == 'offline':
            if self.clustering_algo == 'ward':
                offw_config = config['offline']['ward']
                if offw_config['dynamic_channels']:
                    self.cluster_dist_threshold = offw_config['cluster_dist_threshold']
                    self.cluster_lambda = offw_config['cluster_lambda']
            else:
                raise NotImplementedError
            self.cluster_representation = offw_config['cluster_representation']
            self.cluster_selection_type = offw_config['cluster_selection_type']
            self.dynamic_channels = offw_config['dynamic_channels'] if 'dynamic_channels' in offw_config else True
            if self.cluster_representation == "weighted_mean":
                self.weight_type = offw_config['weights']
                self.weights_lambda = offw_config['weights_lambda']
        else:
            raise NotImplementedError

    def map_channel(self, doc_ids):
        channels = vmap_idx(doc_ids, self.id_index_map)
        valid_idx = (channels >= 0)
        channels[valid_idx] = self.index_channel_map[channels[valid_idx]]
        return channels

    def get_channels(self, activity, embeddings,
                     n_channels=10, doc_ids=None):
        '''
            Get channels from activity and embeddings

            activity - scipy sparse matrix representing user activity
            embeddings - User activity embeddings
            n_channels - number of novel channels returned.
                        It should be equal to max_novel as these many
                        novel behaviour would be returned as novel data
            doc_ids - list of doc indices corresponding to embeddings
        '''
        if self.cluster_mode == 'offline':
            if self.clustering_algo == 'ward':
                return self.get_channels_offline(activity, embeddings, n_channels)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def get_channels_offline(self, activity, embeddings,
                             n_channels=10, doc_ids=None):
        '''
            Get offline channels from activity and embeddings
        '''
        novel_indices = []
        novel_data = []
        novel_indptr = [0]
        all_channels_importance = []
        for i in range(activity.shape[0]):
            user_indices = activity.indices[activity.indptr[i]:activity.indptr[i+1]]
            if (len(user_indices) == 0):
                novel_indptr.append(novel_indptr[-1])
                continue
            elif (len(user_indices) == 1):  # No clustering required
                user_embs = embeddings[user_indices]
                if 'selected_channels' not in locals():
                    selected_channels = user_embs
                    novel_indices.append(0)
                    novel_data.append(0)
                    novel_indptr.append(1)
                    all_channels_importance.append(1.0)
                else:
                    selected_channels = np.vstack([selected_channels, user_embs])
                    novel_indices.append(selected_channels.shape[0]-1)
                    novel_data.append(selected_channels.shape[0]-1)
                    novel_indptr.append(selected_channels.shape[0])
                    all_channels_importance.append(1.0)
                continue

            if self.initial_novel != -1:
                user_indices = user_indices[-self.initial_novel:]
            user_embs = embeddings[user_indices]

            if self.clustering_algo == "ward":
                if self.dynamic_channels:
                    cluster_res = AgglomerativeClustering(distance_threshold=self.cluster_dist_threshold,
                                                          n_clusters=None).fit(user_embs)
                    selected_labels = cluster_res.labels_
                    n_clusters = cluster_res.n_clusters_
                else:
                    cluster_res = AgglomerativeClustering(n_clusters=min(n_channels, len(user_embs))).fit(user_embs)
                    selected_labels = cluster_res.labels_
                    n_clusters = cluster_res.n_clusters_
            else:
                raise NotImplementedError

            clusters = {}
            time_recency = {}
            weights = {}
            stack = deque()

            for j, (A, B) in enumerate(zip(selected_labels, user_embs)):
                if A in clusters:
                    clusters[A] = np.vstack([clusters[A], B])
                    time_recency[A].append(len(selected_labels)-j)
                else:
                    clusters[A] = B
                    time_recency[A] = [len(selected_labels)-j]

            if self.dynamic_channels:
                if self.cluster_selection_type == 'imp_wt':
                    channel_importance = []
                    for j in range(n_clusters):
                        imp = 0
                        for t in time_recency[j]:
                            imp += math.e**(-1*(self.cluster_lambda*t))
                        channel_importance.append(imp)
                    selected_channels_indices = np.argsort(channel_importance)[::-1]
                    # We have channels according to importance. Now select only most important n_channels.
                    # If total clusters are less than n_channels, then pass all of those
                    if n_clusters > n_channels:
                        selected_channels_indices = selected_channels_indices[0:n_channels]
                        channel_importance = list(np.array(channel_importance)[selected_channels_indices])
            else:
                selected_channels_indices = selected_labels

            for j in selected_channels_indices:
                if self.cluster_representation == 'mean':
                    channel_select = np.mean(clusters[j], axis=0) if len(clusters[j].shape) > 1 else clusters[j]
                    orig_shape = channel_select.shape
                    channel_select = normalize(channel_select.reshape(1, -1)).reshape(orig_shape)
                else:
                    raise NotImplementedError

                if 'selected_channels' not in locals():
                    selected_channels = channel_select[np.newaxis, :]
                    novel_indices.append(0)
                    novel_data.append(0)
                else:
                    selected_channels = np.vstack([selected_channels, channel_select])
                    novel_indices.append(selected_channels.shape[0]-1)
                    novel_data.append(selected_channels.shape[0]-1)

            all_channels_importance += channel_importance

            novel_indptr.append(selected_channels.shape[0])

        novel_csr = sp.csr_matrix((novel_data, novel_indices, novel_indptr))
        importance_csr = sp.csr_matrix((all_channels_importance, novel_indices, novel_indptr))
        return novel_csr, selected_channels, importance_csr, [], []


# Given spmat matrix modify it in place to retain last k elements per row
def take_last_k_spmat(spmat, max_elements=30, filter_num_last=0):
    for i in range(spmat.shape[0]):
        spmat.data[spmat.indptr[i]:spmat.indptr[i+1]][:-(max_elements+filter_num_last)] = 0
        if filter_num_last > 0:  # indexing with [-0:] would give full array
            spmat.data[spmat.indptr[i]:spmat.indptr[i+1]][-filter_num_last:] = 0
    spmat.eliminate_zeros()
    
class ActivitySelector():
    '''
    Processes user activity. (channel assignment, etc)
        config: config setting all the parameters needed
    '''
    def __init__(self, config):
        self.type = config['type']  # in ['novel_events', 'channels'(global channels)]
        self.max_novel = config['max_novel']
        self.dedup_novel = config['dedup_novel']
        self.max_history = config['max_history']
        self.history_novel_overlap = config['history_novel_overlap']

        self.userc = Channels(config['uc_config'])
        self.initial_novel = config['uc_config']['initial_novel']

    def __call__(self, params):
        history = params['history']
        doc_map = params['doc_map']
        doc_emb = params['doc_emb']

        # Trimming data to the batch
        # history.indices contains doc ids hence doc_indices is a array of doc ids in the batch
        doc_indices = np.array(list(set(history.indices)))
        # Returns updated doc_id->index map and embeddings that contains docs only in the batch
        batch_doc_map, batch_doc_embs, batch_doc_ids = get_indices_map_embs(doc_indices,
                                                                            doc_map,
                                                                            doc_emb,
                                                                            return_indices=True)
        # history.indices are now indices of embeddings/trainining_points not ids
        history.indices = vmap_idx(history.indices, batch_doc_map).astype(np.int32)
        # For indices<0 (doc ids not found in embeddings), make data as 0
        history.data[history.indices < 0] = 0
        # Clean 0 data, as entries not present in sparse matrix as 0 by default
        history.eliminate_zeros()

        novel = history.copy()
        # Getting channels embeddings and indices by user wise channels
        # Channel indices will be returned if the channel embeddings is the same as novel events
        novel_channels, channel_embs, \
            novel_importance, channel_indices, global_clusters = self.userc.get_channels(novel,
                                                                                         batch_doc_embs,
                                                                                         n_channels=self.max_novel,
                                                                                         doc_ids=batch_doc_ids)
        take_last_k_spmat(history, self.max_history, 0 if self.history_novel_overlap else self.max_novel)
        # take_last_k_only_history(history, self.max_history, self.initial_novel, self.history_novel_overlap)

        inverse_batch_doc_map = {}
        if len(doc_indices) > 0:
            novel_offset = batch_doc_embs.shape[0]
            novel_channels.indices += novel_offset
            # batch_doc_map is not updated if channel_indices==-1 because channels do not correspond to a document
            if len(channel_indices) != 0:
                inverse_batch_doc_map = {v: k for k, v in batch_doc_map.items()}
                for ind in range(0, len(channel_indices)):
                    # old_index: doc_id mapping
                    old_index = int(channel_indices[ind])
                    new_index = int(novel_channels.indices[ind])
                    doc_id = inverse_batch_doc_map[old_index]
                    inverse_batch_doc_map[new_index] = doc_id
                    batch_doc_map[doc_id] = new_index
            batch_doc_embs = np.vstack((batch_doc_embs, channel_embs))
        else:
            batch_doc_map = None
            batch_doc_embs = channel_embs

        return history, novel_channels, batch_doc_map, inverse_batch_doc_map, batch_doc_embs, \
            novel_importance, global_clusters
