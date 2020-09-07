import tensorflow.compat.v1 as tf
from tensorflow.contrib import slim
import numpy as np
from sklearn.metrics import accuracy_score
from models.utils import sparse_dropout

spdot = tf.sparse_tensor_dense_matmul
dot = tf.matmul
tf.set_random_seed(15)
flags = tf.app.flags
FLAGS = flags.FLAGS


class LATGCN:
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=-1, with_reg=False):
        """ Create a Graph Convolutional Network model in Tensorflow with one hidden layer.

        Parameters
        ----------
        sizes : list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]
        An : sp.sparse.csr.csr_matrix, shape [N, N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.
        X_obs : sp.sparse.csr.csr_matrix, shape [N, D]
            The node features.
        name : str, optional
            Name of the network. Default is "".
        with_relu : bool, optional
            Whether there a nonlinear activation function (ReLU) is used.
            If False, there will also be no bias terms, no regularization and no dropout.
            Default is False.
        params_dict : dict, optional
            Dictionary containing other model parameters.
        gpu_id : int or None, optional
            The GPU ID to be used by Tensorflow. If None, CPU will be used. Default is 0.
        seed : int, optional
            Random initialization for reproducibility. Will be ignored if it is -1. Default is -1.
        with_reg : bool, optional
            Whether to use the regularized model. Default is False.
        """

        self.graph = tf.Graph()
        if seed > -1:
            tf.set_random_seed(seed)

        if An.format != "csr":
            An = An.tocsr()
        tf.set_random_seed(15)
        self.with_reg = with_reg

        with self.graph.as_default():
            with tf.variable_scope(name) as scope:
                w_init = slim.xavier_initializer
                self.name = name
                self.n_classes = sizes[1]
                self.dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
                if not with_relu:
                    self.dropout = 0.
                self.learning_rate = params_dict['learning_rate'] if 'learning_rate' in params_dict else 0.01
                self.weight_decay = params_dict['weight_decay'] if 'weight_decay' in params_dict else 5e-4
                self.N, self.D = X_obs.shape
                self.node_ids = tf.placeholder(tf.int32, [None], 'node_ids')
                self.node_labels = tf.placeholder(tf.int32, [None, sizes[1]], 'node_labels')
                # bool placeholder to turn on dropout during training
                self.training = tf.placeholder_with_default(False, shape=())
                self.An = tf.SparseTensor(np.array(An.nonzero()).T, An[An.nonzero()].A1, An.shape)
                self.An = tf.cast(self.An, tf.float32)
                self.X_sparse = tf.SparseTensor(np.array(X_obs.nonzero()).T, X_obs[X_obs.nonzero()].A1, X_obs.shape)
                self.X_dropout = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                                (int(self.X_sparse.values.get_shape()[0]),))
                # only use drop-out during training
                self.X_comp = tf.cond(self.training,
                                      lambda: self.X_dropout,
                                      lambda: self.X_sparse) if self.dropout > 0. else self.X_sparse

                self.W1 = slim.variable('W1', [self.D, sizes[0]], tf.float32, initializer=w_init())
                self.b1 = slim.variable('b1', dtype=tf.float32, initializer=tf.zeros(sizes[0]))

                self.h1 = spdot(self.An, spdot(self.X_comp, self.W1))

                if with_relu:
                    self.h1 = tf.nn.relu(self.h1 + self.b1)

                self.h1_dropout = tf.nn.dropout(self.h1, 1 - self.dropout)

                self.h1_comp = tf.cond(self.training,
                                       lambda: self.h1_dropout,
                                       lambda: self.h1) if self.dropout > 0. else self.h1

                self.W2 = slim.variable('W2', [sizes[0], sizes[1]], tf.float32, initializer=w_init())
                self.b2 = slim.variable('b2', dtype=tf.float32, initializer=tf.zeros(sizes[1]))

                self.logits = spdot(self.An, dot(self.h1_comp, self.W2))
                if with_relu:
                    self.logits += self.b2
                self.logits_gather = tf.gather(self.logits, self.node_ids)

                self.predictions = tf.nn.softmax(self.logits_gather)

                self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits_gather,
                                                                                labels=self.node_labels)
                self.train_loss = tf.reduce_mean(self.loss_per_node)

                # add a regularizer
                if with_reg:
                    self.zeta = tf.Variable(np.random.randn(An.shape[0], sizes[0]), dtype=tf.float32,
                                            constraint=lambda x: tf.clip_by_norm(x, FLAGS.eta, axes=1))
                    h1p = self.h1 + self.zeta
                    h2p = spdot(self.An, dot(h1p, self.W2))
                    self.reg = tf.linalg.norm(h2p - self.logits)
                    # self.reg = tf.linalg.norm(spdot(self.An, dot(self.zeta, self.W2)))
                    self.loss = self.train_loss + self.reg * FLAGS.gamma
                # weight decay only on the first layer, to match the original implementation
                if with_relu:
                    self.loss = self.train_loss + self.weight_decay * \
                        tf.add_n([tf.nn.l2_loss(v)
                                  for v in [self.W1, self.b1]])
                else:
                    self.loss = self.train_loss

                var_l = [self.W1, self.W2]
                if with_relu:
                    var_l.extend([self.b1, self.b2])
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
                self.train_w = self.train_op.minimize(
                    self.loss, var_list=var_l)

                if self.with_reg:
                    self.train_zeta = self.train_op.minimize(-self.reg, var_list=[self.zeta])
                self.varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.local_init_op = tf.variables_initializer(self.varlist)

                if gpu_id is None:
                    config = tf.ConfigProto(device_count={'GPU': 0})
                else:
                    gpu_options = tf.GPUOptions(visible_device_list='{}'.format(gpu_id), allow_growth=True)
                    config = tf.ConfigProto(gpu_options=gpu_options)

                self.session = tf.InteractiveSession(config=config)
                self.init_op = tf.global_variables_initializer()
                self.session.run(self.init_op)

    def convert_varname(self, vname, to_namespace=None):
        """ Utility function that converts variable names to the input namespace.

        Parameters
        ----------
        vname : str
            The variable name.
        to_namespace : str
            The target namespace.

        Returns
        -------
        str
            New name for variable after converting.
        """
        namespace = vname.split("/")[0]
        if to_namespace is None:
            to_namespace = self.name
        return vname.replace(namespace, to_namespace)

    def set_variables(self, var_dict):
        """ Set the model's variables to those provided in var_dict.
        This is e.g. used to restore the best seen parameters after training with patience.

        Parameters
        ----------
        var_dict: dict
            Dictionary of the form {var_name: var_value} to assign the variables in the model.
        """
        with self.graph.as_default():
            if not hasattr(self, 'assign_placeholders'):
                self.assign_placeholders = {v.name: tf.placeholder(
                    v.dtype, shape=v.get_shape()) for v in self.varlist}
                self.assign_ops = {v.name: tf.assign(v, self.assign_placeholders[v.name])
                                   for v in self.varlist}
            to_namespace = list(var_dict.keys())[0].split("/")[0]
            self.session.run(list(self.assign_ops.values()), feed_dict={val: var_dict[self.convert_varname(key, to_namespace)]
                                                                        for key, val in self.assign_placeholders.items()})

    def train(self, split_train, split_val, Z_obs, patience=30, n_iters=200, print_info=False):
        """ Train the GCN model on the provided data.

        Parameters
        ----------
        split_train : np.ndarray, shape [n_train,]
            The indices of the nodes used for training
        split_val : np.ndarray, shape [n_val,]
            The indices of the nodes used for validation.
        Z_obs : np.ndarray, shape [N,k]
            All node labels in one-hot form (the labels of nodes outside of split_train and split_val will not be used.
        patience: int, optional
            After how many steps without improvement of validation error to stop training.
            Default is 30.
        n_iters: int, optional
            Maximum number of iterations (usually we hit the patience limit earlier).
            Default is 200.
        print_info: bool, optional
            Flag to print converge information.
            Default is False.
        """
        varlist = self.varlist
        self.session.run(self.local_init_op)
        early_stopping = patience

        best_performance = 0
        patience = early_stopping

        feed = {self.node_ids: split_train,
                self.node_labels: Z_obs[split_train]}
        if hasattr(self, 'training'):
            feed[self.training] = True
        for it in range(n_iters):
            if self.with_reg:
                for k in range(20):
                    self.session.run(self.train_zeta)
            _loss, _ = self.session.run([self.loss, self.train_w], feed)

            # f1_micro, f1_macro = self.eval_class(split_val, np.argmax(Z_obs, 1))
            acc_val = self.eval_class(split_val, np.argmax(Z_obs, 1))
            # perf_sum = f1_micro + f1_macro
            if acc_val > best_performance:
                best_performance = acc_val
                patience = early_stopping
                # var dump to memory is much faster than to disk using checkpoints
                var_dump_best = {v.name: v.eval(self.session) for v in varlist}
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                break
        if print_info:
            print('converged after {} iterations'.format(it - patience))
        # Put the best observed parameters back into the model
        self.set_variables(var_dump_best)

    def eval_class(self, ids_to_eval, z_obs):
        """ Evaluate the model's classification performance.

        Parameters
        ----------
        ids_to_eval : np.ndarray
            The indices of the nodes whose predictions will be evaluated.
        z_obs : np.ndarray
            The labels of the nodes in ids_to_eval.

        Returns
        -------
        float
            Accuracy score in evaluation.
        """
        test_pred = self.predictions.eval(session=self.session, feed_dict={
                                          self.node_ids: ids_to_eval}).argmax(1)
        test_real = z_obs[ids_to_eval]
        return accuracy_score(test_real, test_pred)
