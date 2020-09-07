from models import utils, latgcn
from models import nettack as ntk
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm
# import scipy.sparse as sp
import os

# set logging level
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# set this to your desired GPU ID if you want to use GPU computations (only for the GCN/surrogate training)
gpu_id = 0

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_bool('reg', False, 'Toggle the regularizer.')
flags.DEFINE_float('eta', 0.1, 'Row constraints on perturbation of latent layer.')
flags.DEFINE_float('gamma', 0.1, 'Factor of regularizer.')
flags.DEFINE_float('train_share', 0.1, 'Percent of testing size.')
flags.DEFINE_bool('output', False, 'Toggle the output.')

#
FLAGS.eta = 0 if not FLAGS.reg else FLAGS.eta
FLAGS.gamma = 0 if not FLAGS.reg else FLAGS.gamma


seed = 15
np.random.seed(seed)
tf.set_random_seed(seed)


def main():
    # build the surrogate model
    s_model = latgcn.LATGCN(sizes, An_mat, X_mat, with_relu=False, name="surrogate", gpu_id=gpu_id)
    s_model.train(train_idx, val_idx, Z_mat, print_info=False)
    W1 = s_model.W1.eval(session=s_model.session)
    W2 = s_model.W2.eval(session=s_model.session)

    # release the resources
    s_model.session.close()
    del(s_model)

    # Setup Nettack
    counter = 0
    success_attack = 0
    losses_before = []
    losses_after = []

    # pick the poisoning or evasion, take 100 nodes from test set
    split_to_attack = np.random.choice(test_idx, 100)

    for u in tqdm(split_to_attack, desc='attack node'):
        counter += 1
        nettack = ntk.Nettack(A_mat, X_mat, z_vec, W1, W2, u, verbose=False)

        # hyper parameters
        direct_attack = True
        n_influencers = 1 if direct_attack else 5
        # How many perturbations to perform. Default: Degree of the node
        n_perturbations = 1
        perturb_features = False
        perturb_structure = True

        # Evasion attack
        nettack.reset()
        nettack.attack_surrogate(n_perturbations,
                                 perturb_structure=perturb_structure,
                                 perturb_features=perturb_features,
                                 direct=direct_attack,
                                 n_influencers=n_influencers)

        gcn_before = latgcn.LATGCN(sizes, An_mat, X_mat, "gcn_orig", gpu_id=gpu_id)
        # ### Train GCN without perturbations
        gcn_before.train(train_idx, val_idx, Z_mat, print_info=False)

        probs_before_attack = gcn_before.predictions.eval(session=gcn_before.session,
                                                          feed_dict={gcn_before.node_ids: [u]})[0]
        # test_acc = gcn_before.eval_class(split_unlabeled, np.argmax(_Z_obs, 1))

        preds_test_before = gcn_before.predictions.eval(session=gcn_before.session,
                                                        feed_dict={gcn_before.node_ids: test_idx})
        loss_train_before = gcn_before.session.run(gcn_before.loss,
                                                   feed_dict={gcn_before.node_ids: train_idx,
                                                              gcn_before.node_labels: Z_mat[train_idx]})
        gcn_before.session.close()
        # class_distrs_clean.append(probs_before_attack)
        # best_second_class_before = (probs_before_attack - 1000*_Z_obs[nettack.u]).argmax()
        # margin_before = probs_before_attack[_z_obs[nettack.u]] - probs_before_attack[best_second_class_before]

        # ### Train GCN with perturbations
        gcn_after = latgcn.LATGCN(sizes, nettack.adj_preprocessed, nettack.X_obs.tocsr(),
                                  "gcn_after", gpu_id=gpu_id, with_reg=FLAGS.reg)
        gcn_after.train(train_idx, val_idx, Z_mat, print_info=False)
        probs_after_attack = gcn_after.predictions.eval(session=gcn_after.session, feed_dict={
                                                        gcn_after.node_ids: [nettack.u]})[0]
        preds_test_after = gcn_after.predictions.eval(session=gcn_after.session,
                                                      feed_dict={gcn_after.node_ids: test_idx})
        loss_after = gcn_after.session.run(gcn_after.loss, feed_dict={
                                           gcn_after.node_ids: train_idx, gcn_after.node_labels: Z_mat[train_idx]})
        # test_acc = gcn_after.eval_class(split_unlabeled, _z_obs)
        # print("{:.4f}".format(test_acc))
        # best_second_class_after = (probs_after_attack - 1000*_Z_obs[nettack.u]).argmax()
        # margin_after = probs_after_attack[_z_obs[nettack.u]] - probs_after_attack[best_second_class_after]

        # calculating the success rate
        if z_vec[nettack.u] == np.argmax(probs_before_attack) and np.argmax(probs_before_attack) != np.argmax(probs_after_attack):
            losses_before.append(loss_train_before)
            losses_after.append(loss_after)
            success_attack += 1

        # inspect the statistics of the prediction
        correct = 0
        correct_after = 0
        correct_before = 0
        incorrect = 0
        for i, n in enumerate(test_idx):
            if z_vec[n] == preds_test_before[i].argmax() and z_vec[n] == preds_test_after[i].argmax():
                correct += 1
            elif z_vec[n] != preds_test_before[i].argmax() and z_vec[n] == preds_test_after[i].argmax():
                correct_after += 1
            elif z_vec[n] == preds_test_before[i].argmax() and z_vec[n] != preds_test_after[i].argmax():
                correct_before += 1
            elif z_vec[n] != preds_test_before[i].argmax() and z_vec[n] != preds_test_after[i].argmax():
                incorrect += 1
        # print("{},{},{},{},{}".format(
        #     len(split_unlabeled), correct, correct_before, correct_after, incorrect
        # ))

        # free memory
        gcn_after.session.close()
        del(nettack)
        del(gcn_before)
        del(gcn_after)
    # formatted outptut at the end
    # print(losses_before, losses_after)

    def aver_loss(loss):
        if len(loss) == 0:
            return 0
        return sum(loss) / len(loss)

    print('{},{:.2f},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{},{:.3f}'.format(
        FLAGS.dataset,
        FLAGS.train_share,
        FLAGS.reg,
        FLAGS.eta,
        FLAGS.gamma,
        aver_loss(losses_before),
        aver_loss(losses_after),
        counter,
        success_attack,
        success_attack / counter))


if __name__ == "__main__":

    # Load network, basic setup
    A_mat, X_mat, z_vec = utils.load_npz(FLAGS.dataset)
    A_mat = A_mat + A_mat.T
    A_mat[A_mat > 1] = 1
    lcc = utils.largest_connected_components(A_mat)

    A_mat = A_mat[lcc][:, lcc]

    assert np.abs(A_mat - A_mat.T).sum() == 0, "Input graph is not symmetric"
    assert A_mat.max() == 1 and len(np.unique(A_mat[A_mat.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert A_mat.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    X_mat = X_mat[lcc].astype('float32')
    z_vec = z_vec[lcc]
    _N = A_mat.shape[0]
    K = z_vec.max() + 1
    Z_mat = np.eye(K)[z_vec]
    An_mat = utils.preprocess_graph(A_mat)
    sizes = [16, K]
    degrees = A_mat.sum(0).A1

    val_size = int(_N * FLAGS.train_share)
    train_size = int(_N * 0.1)
    # train_size = _N - unlabeled_size - val_size
    unlabeled_size = _N - train_size - val_size

    train_idx, val_idx, test_idx = utils.train_val_test_split_tabular(np.arange(_N),
                                                                      train_size=train_size,
                                                                      val_size=val_size,
                                                                      test_size=unlabeled_size,
                                                                      stratify=z_vec)

    main()
