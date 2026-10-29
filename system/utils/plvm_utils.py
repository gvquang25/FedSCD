import copy
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn import metrics

ANN_THRESHOLD = 70000


def clust_rank(mat, initial_rank=None, distance='cosine'):
    s = mat.shape[0]
    if initial_rank is not None:
        orig_dist = []
    else:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s))
    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T

    A = A.tolil()
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None and len(orig_dist) > 0:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean(M, u):
    s = M.shape[0]
    un, u_idx, nf = np.unique(u, return_inverse=True, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u_idx)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.shape)
    a[x, y] = 1
    return a


def req_numclust(c, data, req_clust, distance):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance)
        adj = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
    return c_


def FINCH(data, initial_rank=None, req_clust=None, distance='cosine', ensure_early_exit=True, verbose=False):
    """
    FINCH clustering algorithm (CVPR 2019, Sarfraz et al.).
    Parameter-free clustering based on first integer nearest-neighbor relations.
    """
    data = data.astype(np.float32)
    s = data.shape[0]
    if s <= 1:
        return np.zeros((s, 1), dtype=np.int64), [1], None

    min_sim = None
    adj, orig_dist = clust_rank(data, initial_rank, distance)
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if verbose:
        print('Partition 0: {} clusters'.format(num_clust))

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    exit_clust = 2
    c_ = c
    k = 1
    num_clust = [num_clust]

    while exit_clust > 1:
        if mat.shape[0] <= 1:
            break
        adj, orig_dist = clust_rank(mat, initial_rank, distance)
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)

        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_)) if c.ndim > 1 else np.column_stack((c[:, None], c_[:, None]))
        exit_clust = num_clust[-2] - num_clust_curr

        if num_clust_curr == 1 or exit_clust < 1:
            if len(num_clust) > 1:
                num_clust = num_clust[:-1]
                c = c[:, :-1]
            break

        if verbose:
            print('Partition {}: {} clusters'.format(k, num_clust[k]))
        k += 1

    if c.ndim == 1:
        c = c[:, None]

    if req_clust is not None:
        if req_clust not in num_clust:
            ind = [i for i, v in enumerate(num_clust) if v >= req_clust]
            req_c = req_numclust(c[:, ind[-1]], data, req_clust, distance)
        else:
            req_c = c[:, num_clust.index(req_clust)]
    else:
        req_c = None

    return c, num_clust, req_c


def average_protos(protos_dict):
    """
    Average the representations for each class label:
    Input: {label: [proto1, proto2, ...]}
    Returns: {label: np.ndarray (mean prototype)}
    """
    agg_protos = {}
    for label, proto_list in protos_dict.items():
        if len(proto_list) > 0:
            proto_arr = np.stack(proto_list)
            agg_protos[label] = np.mean(proto_arr, axis=0)
    return agg_protos


def cluster_protos_finch(protos_label_dict):
    """
    Clusters representations for each class label using FINCH:
    Input: {label: [proto1, proto2, ...]}
    Returns:
        agg_protos: {label: [cluster_proto_center_1, cluster_proto_center_2, ...]}
        avg_num_clusters: float
    """
    agg_protos = {}
    num_p = 0
    if not protos_label_dict:
        return agg_protos, 0.0

    for label, proto_list in protos_label_dict.items():
        if len(proto_list) == 0:
            continue
        proto_arr = np.stack(proto_list)
        if len(proto_arr) <= 1:
            agg_protos[label] = [np.mean(proto_arr, axis=0)]
            num_p += 1
            continue

        c, num_clust, _ = FINCH(proto_arr, initial_rank=None, req_clust=None, distance='cosine',
                                ensure_early_exit=False, verbose=False)
        if c.ndim == 1:
            c = c[:, None]

        class_cluster_array = c[:, -1]
        unique_clusters = np.unique(class_cluster_array).tolist()
        agg_selected_proto = []

        for cluster_index in unique_clusters:
            selected_indices = np.where(class_cluster_array == cluster_index)[0]
            selected_proto_list = proto_arr[selected_indices]
            cluster_proto_center = np.mean(selected_proto_list, axis=0)
            agg_selected_proto.append(cluster_proto_center)

        agg_protos[label] = agg_selected_proto
        num_p += len(unique_clusters)

    avg_num_clusters = (num_p / len(protos_label_dict)) if len(protos_label_dict) > 0 else 0.0
    return agg_protos, avg_num_clusters


def local_cluster_collect(local_cluster_protos_dict):
    """
    Collects all local cluster prototypes across clients into a global dictionary:
    Input: {client_id: {label: [cluster_proto_1, cluster_proto_2, ...]}}
    Returns: {label: [cluster_proto_1, cluster_proto_2, ...]}
    """
    global_collected_protos = {}
    for idx, cluster_protos_label in local_cluster_protos_dict.items():
        for label, cluster_protos_list in cluster_protos_label.items():
            if label not in global_collected_protos:
                global_collected_protos[label] = []
            for p in cluster_protos_list:
                global_collected_protos[label].append(p)
    return global_collected_protos


def proto_aggregation_cluster(global_cluster_protos):
    """
    Averages cluster prototypes per label to construct global average prototypes:
    Input: {label: [cluster_proto_1, cluster_proto_2, ...]}
    Returns: {label: np.ndarray (class-wise average prototype)}
    """
    agg_protos_label = {}
    for label, proto_list in global_cluster_protos.items():
        if len(proto_list) > 0:
            proto_arr = np.stack(proto_list)
            agg_protos_label[label] = np.mean(proto_arr, axis=0)
    return agg_protos_label
