from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import os
import os.path as osp
import scipy.sparse as sp
import tensorflow.compat.v1 as tf


def xavier_init(size):
    """ The initiation from the Xavier's paper.
        ref: Understanding the difficulty of training deep feedforward neural
            networks, Xavier Glorot, Yoshua Bengio, AISTATS 2010.
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    Parameters
    ----------
    size : tuple, shape (N, h)
        Size of the variable.

    Returns
    -------
    tf.Tensor
        An initialized variable
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sparse_dropout(x, keep_prob, noise_shape):
    """ Dropout for sparse tensors.

    Parameters
    ----------
    x : tf.sparse.SparseTensor
        SparseTensor as input.
    keep_prob : float
        keep_prob
    noise_shape : tuple

    Returns
    -------
    tf.sparse.SparseTensor : sparse tensor after applying dropout.

    Notes
    -----
    Recent tf.nn.dropout will use `rate` instead of `keep_prob`.

    See Also
    --------
    tf.compat.v1.nn.dropout : Computes dropout. (deprecated arguments)
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def load_npz(dataset):
    """ Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.

    Returns
    -------
    adj_matrix : sp.csr.csr_matrix, shape (N, N)
        Adjacency matrix in sparse matrix format.
    attr_matrix : sp.csr.csr_matrix, shape (N, D)
        Attribute matrix in sparse matrix format.
    labels : np.ndarray, shape (N, )
        Labels of all nodes.

    Notes
    -----
    * If there is no feature matrix, use identity matrix instead.
    """
    import requests
    url = 'https://www.cs.uic.edu/~hjin/data/{}.npz'.format(dataset.lower())
    filename = "{}.npz".format(dataset)
    data_root = osp.expanduser('~')
    dir_path = osp.join(data_root, 'web', dataset)
    file_path = osp.join(dir_path, filename)
    # download file if not exists
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
    if not osp.exists(file_path):
        r = requests.get(url, stream=True)
        content_size = int(r.headers['Content-Length']) / 1024
        with open(file_path, 'wb') as outfile:
            for data in tqdm(iterable=r.iter_contentls(1024),
                             total=content_size,
                             desc="Download {}".format(dataset)):
                outfile.write(data)
    # load from npz
    with np.load(file_path, allow_pickle=True) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix(
            (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
            shape=loader['adj_shape'])
        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix(
                (loader['attr_data'], loader['attr_indices'],
                 loader['attr_indptr']),
                shape=loader['attr_shape'])
        else:
            attr_matrix = sp.eye(adj_matrix.shape[0], format='csr')
        labels = loader.get('labels')
    return adj_matrix, attr_matrix, labels


def largest_connected_components(adj, n_components=1):
    """ (Deprecated) Select the largest connected components in the graph.

    Parameters
    ----------
    adj : sp.csr.csr_matrix, shape (N, N)
        Input graph as sparse matrix.
    n_components : int, optional
        Number of largest connected components to keep.
        Default is 1.

    Returns
    -------
    nodes_to_keep : list
        List of nodes in largest connected component.
    """
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    # reverse order to sort descending
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return nodes_to_keep


def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """ Split the arrays or matrices into random train, validation and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels.
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.
    """
    # DEBUG: fix the error when sum(train_size + test_size) != samples
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    # train_sample = len(idx) * train_size
    # val_sample = len(idx) * val_size
    # test_sample = len(idx) - train_sample - val_sample
    idx_train_and_val, idx_test = train_test_split(
        idx,
        random_state=random_state,
        train_size=train_size + val_size,
        test_size=test_size,
        stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]
        idx_train, idx_val = train_test_split(
            idx_train_and_val,
            random_state=random_state,
            train_size=train_size,
            test_size=val_size,
            stratify=stratify)

    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result


def preprocess_graph(adj):
    """ Return the normalized laplacian matrix
        normalized_laplacian = D^{-1/2} (D-A)D^{-1/2}

    Parameters
    ----------
    adj: a sparse matrix represents the adjacency matrix

    Returns
    -------
    adj_normalized: a sparse matrix represents the normalized laplacian
        matrix
    """
    adj_ = adj + 1 * sp.eye(adj.shape[0])
    rowsum = adj_.sum(1).A1
    D_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
    adj_normalized = D_inv_sqrt @ adj_ @ D_inv_sqrt
    return adj_normalized.tocsr()


def sp_matrix_to_sp_tensor(M):
    """ Convert a sparse matrix to a SparseTensor

    Parameters
    ----------
    M : sp.csr.csr_matrix
        Input sparse matrix.

    Returns
    -------
    X : tf.sparse.SparseTensor

    See Also
    --------
    tf.SparseTensor, scipy.sparse.csr_matrix
    """
    row, col = M.nonzero()
    X = tf.SparseTensor(np.mat([row, col]).T, M.data, M.shape)
    X = tf.cast(X, tf.float32)
    return X
