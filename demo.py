# coding: utf-8

from models import utils, LATGCN
from models import nettack as ntk
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.metrics import *

# set this to your desired GPU ID if you want to use GPU computations (only for the GCN/surrogate training)
gpu_id = None 

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
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


def run_model(model, A, X, u, with_reg=FLAGS.reg):
    """ Run models with different settings
    
    Parameters
    ----------
    model : a GCN model
    A : sparse matrix
    X : sparse matrix
    u : int
        target node
    with_reg : bool
        flag of using regularizer

    Returns
    -------
    None

    """
    # create model
    model = LATGCN.LATGCN(sizes, A, X, 'gcn_origin', with_reg=with_reg)
    model.train(split_train, split_val, _Z_obs, print_info=False)

    # eval on node u
    preds_u = model.predictions.eval(model.session, feed_dict={model.node_ids: [u]})[0]
    
    # evaluate accuracy on test set
    acc = model.eval_class(split_unlabeled, _z_obs[split_unlabeled])
    
    # release resources
    # model.session.close()
    model.session.run(model.init_op)
    del(model)
    return preds_u, acc


# @profile
def create_model():

    surrogate_model = LATGCN.LATGCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs, print_info=False)
    W1 =surrogate_model.W1.eval(session=surrogate_model.session)
    W2 =surrogate_model.W2.eval(session=surrogate_model.session)

    # release the resources
    surrogate_model.session.close()
    del(surrogate_model)

    ### Setup Nettack
    counter = 0
    success_attack = 0
    losses_before = []
    losses_after = []
    
    ### pick the poisoning or evasion, take 100 nodes from test set
    split_to_attack = np.random.choice(split_unlabeled, 100)

    for u in tqdm(split_to_attack[:], desc='attack'):
        counter += 1
        nettack = ntk.Nettack(_A_obs, _X_obs, _z_obs, W1, W2, u, verbose=False)

        # hyper parameters
        direct_attack = True
        n_influencers = 1 if direct_attack else 5
        # How many perturbations to perform. Default: Degree of the node
        n_perturbations = 5
        perturb_features = False
        perturb_structure = True

        # Evasion attack
        nettack.reset()
        nettack.attack_surrogate(n_perturbations,  
                    perturb_structure=perturb_structure, 
                    perturb_features=perturb_features, 
                    direct=direct_attack, 
                    n_influencers=n_influencers)

        gcn_before = LATGCN.LATGCN(sizes, _An, _X_obs, "gcn_orig", gpu_id=gpu_id)
        # ### Train GCN without perturbations
        gcn_before.train(split_train, split_val, _Z_obs, print_info=False)

        probs_before_attack = gcn_before.predictions.eval(session=gcn_before.session,
                                                            feed_dict={gcn_before.node_ids: [u]})[0]
        # print(probs_before_attack)
        test_acc = gcn_before.eval_class(split_unlabeled, np.argmax(_Z_obs, 1))
        # print(test_acc)
        # exit()
        preds_test_before = gcn_before.predictions.eval(session=gcn_before.session,
                                                    feed_dict={gcn_before.node_ids: split_unlabeled})
        loss_train_before = gcn_before.session.run(gcn_before.loss, 
                                            feed_dict={gcn_before.node_ids: split_train, 
                                                    gcn_before.node_labels: _Z_obs[split_train]})

        # class_distrs_clean.append(probs_before_attack)
        # best_second_class_before = (probs_before_attack - 1000*_Z_obs[nettack.u]).argmax()
        # margin_before = probs_before_attack[_z_obs[nettack.u]] - probs_before_attack[best_second_class_before]


        # ### Train GCN with perturbations
        gcn_after = LATGCN.LATGCN(sizes, nettack.adj_preprocessed, nettack.X_obs.tocsr(), "gcn_after", gpu_id=gpu_id, with_reg=FLAGS.reg)
        gcn_after.train(split_train, split_val, _Z_obs, print_info=False)
        probs_after_attack = gcn_after.predictions.eval(session=gcn_after.session,feed_dict={gcn_after.node_ids: [nettack.u]})[0]
        preds_test_after = gcn_after.predictions.eval(session=gcn_after.session,
                                                        feed_dict={gcn_after.node_ids: split_unlabeled})
        loss_after = gcn_after.session.run(gcn_after.loss, feed_dict={gcn_after.node_ids:split_train, gcn_after.node_labels: _Z_obs[split_train]})
        test_acc = gcn_after.eval_class(split_unlabeled, _z_obs)
        print("{:.4f}".format(test_acc))
        # best_second_class_after = (probs_after_attack - 1000*_Z_obs[nettack.u]).argmax()
        # margin_after = probs_after_attack[_z_obs[nettack.u]] - probs_after_attack[best_second_class_after]
            
        # calculating the success rate
        if _z_obs[nettack.u] == np.argmax(probs_before_attack) and np.argmax(probs_before_attack) != np.argmax(probs_after_attack):
            losses_before.append(loss_train_before)
            losses_after.append(loss_after)
            success_attack += 1

        # inspect the statistics of the prediction
        correct = 0
        correct_after = 0
        correct_before = 0
        incorrect = 0
        for i, n in enumerate(split_unlabeled):
            if _z_obs[n] == preds_test_before[i].argmax() and _z_obs[n] == preds_test_after[i].argmax():
                correct += 1
            elif _z_obs[n] != preds_test_before[i].argmax() and _z_obs[n] == preds_test_after[i].argmax():
                correct_after += 1
            elif _z_obs[n] == preds_test_before[i].argmax() and _z_obs[n] != preds_test_after[i].argmax():
                correct_before += 1
            elif _z_obs[n] != preds_test_before[i].argmax() and _z_obs[n] != preds_test_after[i].argmax():
                incorrect += 1 
        print("{},{},{},{},{}".format(
            len(split_unlabeled), correct, correct_before, correct_after, incorrect
        ))
        # print(
        #     "counter {:03d}".format(counter),
        #     "succ_attack: {:03d}".format(success_attack),
        #     "succ_rate: {:.4f}".format(success_attack/counter)
        #     )

        # free memory
        # gcn_before.session.close()
        gcn_before.session.run(gcn_before.init_op)
        gcn_after.session.run(gcn_after.init_op)
        del(nettack)
        del(gcn_before)
        del(gcn_after)
    # formatted outptut at the end
    # print(losses_before, losses_after)
    def aver_loss(loss):
        if len(loss) == 0:
            return 0
        return sum(loss)/len(loss)
    
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
        success_attack/counter))


if __name__ == "__main__":

    ### Load network, basic setup
    _A_obs, _X_obs, _z_obs = utils.load_npz('data/{}.npz'.format(FLAGS.dataset))
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)

    _A_obs = _A_obs[lcc][:,lcc]

    assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
    assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
    assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

    _X_obs = _X_obs[lcc].astype('float32')
    _z_obs = _z_obs[lcc]
    _N = _A_obs.shape[0]
    _K = _z_obs.max()+1
    _Z_obs = np.eye(_K)[_z_obs]
    _An = utils.preprocess_graph(_A_obs)
    sizes = [16, _K]
    degrees = _A_obs.sum(0).A1


    val_size = int(_N*FLAGS.train_share)
    train_size = int(_N * 0.1)
    # train_size = _N - unlabeled_size - val_size
    unlabeled_size = _N - train_size - val_size

    split_train, split_val, split_unlabeled = utils.train_val_test_split_tabular(np.arange(_N),
                                            train_size=train_size,
                                            val_size=val_size,
                                            test_size=unlabeled_size,
                                            stratify=_z_obs)

    create_model()
