from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import cPickle as pickle
import pickle
import shutil
import sys
import time
import pdb

import numpy as np
import tensorflow as tf

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar10.data_utils import read_data
from src.cifar10.general_controller import GeneralController
from src.cifar10.eval_child import GeneralChild

from src.cifar10.micro_controller import MicroController
from src.cifar10.micro_child import MicroChild

sys.argv = sys.argv[:1]  # suppress cmd arguments, use default ones defined below
flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", '%s/../../data/cifar10' % os.path.dirname(os.path.realpath(__file__)), "")
DEFINE_string("output_dir", '%s/../../outputs_6' % os.path.dirname(os.path.realpath(__file__)), "")
DEFINE_string("data_format", "NCHW", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", "macro", "Must be [macro|micro]")

DEFINE_integer("batch_size", 128, "")

DEFINE_integer("num_epochs", 10, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 6, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 36, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 6, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", 10, "for lr schedule")
DEFINE_integer("child_lr_T_mul", 2, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.9, "")
DEFINE_float("child_drop_path_keep_prob", 0.6, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 0.00025, "")
DEFINE_float("child_lr_max", 0.05, "for lr schedule")
DEFINE_float("child_lr_min", 0.0005, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_string("structure_path", "sample_structures6.txt", "")
DEFINE_boolean("child_use_aux_heads", True, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", True, "Use cosine lr schedule")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", 0.0001, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", True, "")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", False, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

class Eval(object):

    def get_ops(self, images, labels):
      """
      Args:
        images: dict with keys {"train", "valid", "test"}.
        labels: dict with keys {"train", "valid", "test"}.
      """

      assert FLAGS.search_for is not None, "Please specify --search_for"

      if FLAGS.search_for == "micro":
        ControllerClass = MicroController
        ChildClass = MicroChild
      else:
        ControllerClass = GeneralController
        ChildClass = GeneralChild

      child_model = ChildClass(
        images,
        labels,
        use_aux_heads=FLAGS.child_use_aux_heads,
        cutout_size=FLAGS.child_cutout_size,
        whole_channels=FLAGS.controller_search_whole_channels,
        num_layers=FLAGS.child_num_layers,
        num_cells=FLAGS.child_num_cells,
        num_branches=FLAGS.child_num_branches,
        fixed_arc=FLAGS.child_fixed_arc,
        out_filters_scale=FLAGS.child_out_filters_scale,
        out_filters=FLAGS.child_out_filters,
        keep_prob=FLAGS.child_keep_prob,
        drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
        num_epochs=FLAGS.num_epochs,
        l2_reg=FLAGS.child_l2_reg,
        data_format=FLAGS.data_format,
        batch_size=FLAGS.batch_size,
        clip_mode="norm",
        grad_bound=FLAGS.child_grad_bound,
        lr_init=FLAGS.child_lr,
        lr_dec_every=FLAGS.child_lr_dec_every,
        lr_dec_rate=FLAGS.child_lr_dec_rate,
        lr_cosine=FLAGS.child_lr_cosine,
        lr_max=FLAGS.child_lr_max,
        lr_min=FLAGS.child_lr_min,
        lr_T_0=FLAGS.child_lr_T_0,
        lr_T_mul=FLAGS.child_lr_T_mul,
        optim_algo="momentum",
        sync_replicas=FLAGS.child_sync_replicas,
        num_aggregate=FLAGS.child_num_aggregate,
        num_replicas=FLAGS.child_num_replicas,
      )

      if FLAGS.child_fixed_arc is None:
        controller_model = ControllerClass(
          search_for=FLAGS.search_for,
          search_whole_channels=FLAGS.controller_search_whole_channels,
          skip_target=FLAGS.controller_skip_target,
          skip_weight=FLAGS.controller_skip_weight,
          num_cells=FLAGS.child_num_cells,
          num_layers=FLAGS.child_num_layers,
          num_branches=FLAGS.child_num_branches,
          out_filters=FLAGS.child_out_filters,
          lstm_size=64,
          lstm_num_layers=1,
          lstm_keep_prob=1.0,
          tanh_constant=FLAGS.controller_tanh_constant,
          op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
          temperature=FLAGS.controller_temperature,
          lr_init=FLAGS.controller_lr,
          lr_dec_start=0,
          lr_dec_every=1000000,  # never decrease learning rate
          l2_reg=FLAGS.controller_l2_reg,
          entropy_weight=FLAGS.controller_entropy_weight,
          bl_dec=FLAGS.controller_bl_dec,
          use_critic=FLAGS.controller_use_critic,
          optim_algo="adam",
          sync_replicas=FLAGS.controller_sync_replicas,
          num_aggregate=FLAGS.controller_num_aggregate,
          num_replicas=FLAGS.controller_num_replicas,
          structure_path=FLAGS.structure_path)

        child_model.connect_controller(controller_model)
        controller_model.build_trainer(child_model)

        controller_ops = {
          "train_step": controller_model.train_step,
          "loss": controller_model.loss,
          "train_op": controller_model.train_op,
          "lr": controller_model.lr,
          "grad_norm": controller_model.grad_norm,
          "valid_acc": controller_model.valid_acc,
          "optimizer": controller_model.optimizer,
          "baseline": controller_model.baseline,
          "entropy": controller_model.sample_entropy,
          "sample_arc": controller_model.sample_arc,
          "sample_arc2": controller_model.sample_arc2,
          "sample_arc3": controller_model.sample_arc3,
          "skip_rate": controller_model.skip_rate,
          "structures": controller_model.structures,
        }
      else:
        assert not FLAGS.controller_training, (
          "--child_fixed_arc is given, cannot train controller")
        child_model.connect_controller(None)
        controller_ops = None

      child_ops = {
        "child": child_model,
        "global_step": child_model.global_step,
        "loss": child_model.loss,
        "train_op": child_model.train_op,
        "lr": child_model.lr,
        "grad_norm": child_model.grad_norm,
        "train_acc": child_model.train_acc,
        "optimizer": child_model.optimizer,
        "num_train_batches": child_model.num_train_batches,
      }

      ops = {
        "child": child_ops,
        "controller": controller_ops,
        "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
        "eval_func": child_model.customized_eval_once,
        "reset_idx": child_model.reset_idx,
        "num_train_batches": child_model.num_train_batches,
      }

      return ops

    def __init__(self):
      if FLAGS.child_fixed_arc is None:
        images, labels = read_data(FLAGS.data_path)
      else:
        images, labels = read_data(FLAGS.data_path, num_valids=0)

      g = tf.Graph()
      with g.as_default():
        self.ops = self.get_ops(images, labels)
        child_ops = self.ops["child"]
        controller_ops = self.ops["controller"]

        saver = tf.train.Saver(max_to_keep=2)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
          FLAGS.output_dir, save_steps=child_ops["num_train_batches"]*10000, saver=saver)

        hooks = [checkpoint_saver_hook]
        if FLAGS.child_sync_replicas:
          sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
          hooks.append(sync_replicas_hook)
        if FLAGS.controller_training and FLAGS.controller_sync_replicas:
          sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
          hooks.append(sync_replicas_hook)

        print("-" * 80)
        print("Starting session")
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.train.SingularMonitoredSession(
            config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir)

    def eval(self, arch_str):
        if type(arch_str) == type(''):
            arch_str = [int(x) for x in arch_str.split()]
        return self.ops["eval_func"](self.sess, "valid", feed_dict={self.ops["controller"]["sample_arc3"]: np.asarray(arch_str)})



def Eval_NN():
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  utils.print_user_flags()

  '''
  # below are for batch evaluation of all arcs defined in the structure_path
  if not FLAGS.structure_path:
    exit()
  with open(FLAGS.structure_path, 'r') as fp:
    lines = fp.readlines()
  lines = [eval(line.strip()) for line in lines]
  structures = []
  for line in lines:
    row = []
    for ele in line:
      row += ele
    structures.append(row) 
  n = len(lines)
  # eval the first structure
  Acc = []
  eva = Eval()
  eva.eval(structures[0])
  eva.eval(structures[1])
  acc = eva.eval(structures[0])
  print(acc)
  pdb.set_trace()
  '''
  eva = Eval()
  return eva


if __name__ == "__main__":
  tf.app.run()
