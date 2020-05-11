# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys

import tensorflow as tf
import numpy as np
import collections

from ops import input_ops

FLAGS = tf.flags.FLAGS

def read_vocab_embs(vocabulary_file, embedding_matrix_file):
  tf.logging.info("Reading vocabulary from %s", vocabulary_file)
  with tf.gfile.GFile(vocabulary_file, mode="r") as f:
    lines = list(f.readlines())
  vocab = [line.strip() for line in lines]

  with open(embedding_matrix_file, "r") as f:
    embedding_matrix = np.load(f)
  tf.logging.info("Loaded embedding matrix with shape %s",
                  embedding_matrix.shape)
  word_embedding_dict = collections.OrderedDict(
      zip(vocab, embedding_matrix))
  return word_embedding_dict

def read_vocab(vocabulary_file):
  tf.logging.info("Reading vocabulary from %s", vocabulary_file)
  with tf.gfile.GFile(vocabulary_file, mode="r") as f:
    lines = list(f.readlines())
  reverse_vocab = [line.strip() for line in lines]
  tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))

  #tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
  # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
  # with 3 arguments.
  word_embedding_dict = collections.OrderedDict(
      zip(reverse_vocab, range(len(reverse_vocab))))
  return word_embedding_dict


class s2v(object):
  """Skip-thoughts model."""

  def __init__(self, config, mode="train", input_reader=None, input_queue=None):
    """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "encode".
      input_reader: Subclass of tf.ReaderBase for reading the input serialized
        tf.Example protocol buffers. Defaults to TFRecordReader.

    Raises:
      ValueError: If mode is invalid.
    """
    if mode not in ["train", "eval", "encode"]:
      raise ValueError("Unrecognized mode: %s" % mode)

    self.config = config
    self.mode = mode
    self.reader = input_reader if input_reader else tf.TFRecordReader()
    self.input_queue = input_queue

    # Initializer used for non-recurrent weights.
    self.uniform_initializer = tf.random_uniform_initializer(
        minval=-FLAGS.uniform_init_scale,
        maxval=FLAGS.uniform_init_scale)

    # Input sentences represented as sequences of word ids. "encode" is the
    # source sentence, "decode_pre" is the previous sentence and "decode_post"
    # is the next sentence.
    # Each is an int64 Tensor with  shape [batch_size, padded_length].
    self.encode_ids = None

    # Boolean masks distinguishing real words (1) from padded words (0).
    # Each is an int32 Tensor with shape [batch_size, padded_length].
    self.encode_mask = None

    # Input sentences represented as sequences of word embeddings.
    # Each is a float32 Tensor with shape [batch_size, padded_length, emb_dim].
    self.encode_emb = None

    # The output from the sentence encoder.
    # A float32 Tensor with shape [batch_size, num_gru_units].
    self.thought_vectors = None

    # The total loss to optimize.
    self.total_loss = None

  def build_inputs(self):

    if self.mode == "encode":
      encode_ids = tf.placeholder(tf.int64, (None, None), name="encode_ids")
      encode_mask = tf.placeholder(tf.int8, (None, None), name="encode_mask")
    else:
      # Prefetch serialized tf.Example protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          FLAGS.input_file_pattern,
          shuffle=FLAGS.shuffle_input_data,
          capacity=FLAGS.input_queue_capacity,
          num_reader_threads=FLAGS.num_input_reader_threads)

      # Deserialize a batch.
      serialized = input_queue.dequeue_many(FLAGS.batch_size)
      encode = input_ops.parse_example_batch(serialized)

      encode_ids = tf.identity(encode.ids, name="encode_ids")
      encode_mask = tf.identity(encode.mask, name="encode_mask")

    self.encode_ids = encode_ids
    self.encode_mask = encode_mask

  def build_word_embeddings(self):

    rand_init = self.uniform_initializer
    self.word_embeddings = []
    self.encode_emb = []
    self.init = None
    for v in self.config.vocab_configs:

      if v.mode == 'fixed':
        if self.mode == "train":
          word_emb = tf.get_variable(
              name=v.name,
              shape=[v.size, v.dim],
              trainable=False)
          embedding_placeholder = tf.placeholder(
              tf.float32, [v.size, v.dim])
          embedding_init = word_emb.assign(embedding_placeholder)

          rand = np.random.rand(1, v.dim)
          word_vecs = np.load(v.embs_file)
          load_vocab_size = word_vecs.shape[0]
          assert(load_vocab_size == v.size - 1)
          word_init = np.concatenate((rand, word_vecs), axis=0)
          self.init = (embedding_init, embedding_placeholder, word_init)
        else:
          word_emb = tf.get_variable(
              name=v.name,
              shape=[v.size, v.dim])

        encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
        self.word_emb = word_emb
        self.encode_emb.extend([encode_emb, encode_emb])

      if v.mode == 'trained':
        for inout in ["", "_out"]:
          word_emb = tf.get_variable(
              name=v.name + inout,
              shape=[v.size, v.dim],
              initializer=rand_init)
          if self.mode == 'train':
            self.word_embeddings.append(word_emb)

          encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
          self.encode_emb.append(encode_emb)

      if v.mode == 'expand':
        for inout in ["", "_out"]:
          encode_emb = tf.placeholder(tf.float32, (
              None, None, v.dim), v.name + inout)
          self.encode_emb.append(encode_emb)
          word_emb_dict = read_vocab_embs(v.vocab_file + inout + ".txt",
              v.embs_file + inout + ".npy")
          self.word_embeddings.append(word_emb_dict)

      if v.mode != 'expand' and self.mode == 'encode':
        word_emb_dict = read_vocab(v.vocab_file)
        self.word_embeddings.extend([word_emb_dict, word_emb_dict])

  def _initialize_cell(self, num_units, cell_type="GRU"):
    if cell_type == "GRU":
      return tf.contrib.rnn.GRUCell(num_units=num_units)
    elif cell_type == "LSTM":
      return tf.contrib.rnn.LSTMCell(num_units=num_units)
    else:
      raise ValueError("Invalid cell type")

  def bow(self, word_embs, mask):
    mask_f = tf.expand_dims(tf.cast(mask, tf.float32), -1)
    word_embs_mask = word_embs * mask_f
    bow = tf.reduce_sum(word_embs_mask, axis=1)
    return bow

  def rnn(self, word_embs, mask, scope, encoder_dim, cell_type="GRU"):

    length = tf.to_int32(tf.reduce_sum(mask, 1), name="length")

    if self.config.bidir:
      if encoder_dim % 2:
        raise ValueError(
            "encoder_dim must be even when using a bidirectional encoder.")
      num_units = encoder_dim // 2
      cell_fw = self._initialize_cell(num_units, cell_type=cell_type)
      cell_bw = self._initialize_cell(num_units, cell_type=cell_type)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=word_embs,
          sequence_length=length,
          dtype=tf.float32,
          scope=scope)
      if cell_type == "LSTM":
        states = [states[0][1], states[1][1]]
      state = tf.concat(states, 1)
    else:
      cell = self._initialize_cell(encoder_dim, cell_type=cell_type)
      outputs, state = tf.nn.dynamic_rnn(
          cell=cell,
          inputs=word_embs,
          sequence_length=length,
          dtype=tf.float32,
          scope=scope)
      if cell_type == "LSTM":
        state = state[1]
    return state

  def build_encoder(self):
    """Builds the sentence encoder.

    Inputs:
      self.encode_emb
      self.encode_mask

    Outputs:
      self.thought_vectors

    Raises:
      ValueError: if config.bidirectional_encoder is True and config.encoder_dim
        is odd.
    """
    names = ["","_out"]
    self.thought_vectors = []
    # print(self.config.encoder)
    # encode_emb = tf.compat.v1.Print(self.encode_emb, [tf.shape(self.encode_emb)], "encode_emb shape: ")
    encode_emb = self.encode_emb
    for i in range(2):
      with tf.variable_scope("encoder" + names[i]) as scope:

        if self.config.encoder == "gru":
          sent_rep = self.rnn(encode_emb[i], self.encode_mask, scope, self.config.encoder_dim, cell_type="GRU")
        elif self.config.encoder == "lstm":
          sent_rep = self.rnn(encode_emb[i], self.encode_mask, scope, self.config.encoder_dim, cell_type="LSTM")
        elif self.config.encoder == 'bow':
          sent_rep = self.bow(encode_emb[i], self.encode_mask)
        else:
          raise ValueError("Invalid encoder")

        if self.config.encoder_norm:
          # sent_rep = tf.compat.v1.Print(sent_rep, [tf.shape(sent_rep)], "sent_rep shape: ")
          sent_rep = tf.nn.l2_normalize(sent_rep, axis=1)

        thought_vectors = tf.identity(sent_rep, name="thought_vectors")
        self.thought_vectors.append(thought_vectors)


  def build_loss(self):
    """Builds the loss Tensor.

    Outputs:
      self.total_loss
    """

    loss_config = self.config.loss_config
    all_sen_embs = self.thought_vectors

    losses = []
    to_log = {}

    print(self.config)
    print(loss_config)

    # Positive pair targets
    # diag = all zeros
    pos_targets_np = np.zeros((FLAGS.batch_size, FLAGS.batch_size))
    ctxt_sent_pos = list(range(-FLAGS.context_size, FLAGS.context_size + 1))
    ctxt_sent_pos.remove(0)
    for ctxt_pos in ctxt_sent_pos:
      pos_targets_np += np.eye(FLAGS.batch_size, k=ctxt_pos)

    pos_targets_np_sum = np.sum(pos_targets_np, axis=1, keepdims=True)
    pos_targets_np = pos_targets_np / pos_targets_np_sum

    # matmul scores
    if FLAGS.dropout:
      mask_shp = [1, self.config.encoder_dim]
      bin_mask = tf.random_uniform(mask_shp) > FLAGS.dropout_rate
      bin_mask = tf.where(bin_mask, tf.ones(mask_shp), tf.zeros(mask_shp))
      src = all_sen_embs[0] * bin_mask
      dst = all_sen_embs[1] * bin_mask
      mm_scores = tf.matmul(src, dst, transpose_b=True)
    else:
      mm_scores = tf.matmul(all_sen_embs[0], all_sen_embs[1], transpose_b=True)

    if loss_config.c != 0:

      c_scores = mm_scores / loss_config.ct

      # Ignore source sentence
      if self.config.encoder_norm:
        nodiag_mask_np = np.ones((FLAGS.batch_size, FLAGS.batch_size), dtype=np.bool)
        np.fill_diagonal(nodiag_mask_np, False)
        nodiag_pos_targets_np = pos_targets_np[nodiag_mask_np].reshape(FLAGS.batch_size, FLAGS.batch_size - 1)
        pos_targets = tf.constant(nodiag_pos_targets_np, dtype=tf.float32)  # still normalized since diag(pos_targets_np) = 0

        nodiag_mask = tf.constant(nodiag_mask_np)
        c_scores = tf.reshape(c_scores[nodiag_mask], [FLAGS.batch_size, FLAGS.batch_size - 1])
      else:
        pos_targets = tf.constant(pos_targets_np, dtype=tf.float32)
        c_scores = tf.matrix_set_diag(c_scores, np.zeros(FLAGS.batch_size))

      c_losses = tf.nn.softmax_cross_entropy_with_logits(
          labels=pos_targets, logits=c_scores)

      to_log['c_loss'] = c_loss = tf.reduce_mean(c_losses)
      losses.append(c_loss * loss_config.c)

    if loss_config.u != 0:
      assert self.config.encoder_norm

      # tril indicies
      mask_np = np.zeros((FLAGS.batch_size, FLAGS.batch_size), dtype=np.bool)
      mask_np[np.tril_indices(FLAGS.batch_size, -1)] = True
      mask = tf.constant(mask_np)

      f_sen = all_sen_embs[0]
      f_scores = tf.matmul(f_sen, f_sen, transpose_b=True)
      uf_loss = tf.log(tf.reduce_mean( tf.exp( (f_scores[mask] - 1) * (2 * loss_config.ut) )))

      g_sen = all_sen_embs[1]
      g_scores = tf.matmul(g_sen, g_sen, transpose_b=True)
      ug_loss = tf.log(tf.reduce_mean( tf.exp( (g_scores[mask] - 1) * (2 * loss_config.ut) )))

      to_log['uf_loss'] = uf_loss
      to_log['ug_loss'] = ug_loss
      losses.append((uf_loss + ug_loss) * loss_config.u)

    if loss_config.a != 0:
      assert self.config.encoder_norm

      pos_targets_norm_np = pos_targets_np / pos_targets_np.sum()
      pos_targets_norm = tf.constant(pos_targets_norm_np, dtype=tf.float32)

      dist_metric = 2 - 2 * mm_scores
      if loss_config.aa != 2:
        # eps = 1e-7
        # dist_metric = (dist_metric + eps) ** (loss_config.aa / 2)
        dist_metric = dist_metric ** (loss_config.aa / 2)

      to_log['a_loss'] = a_loss = tf.reduce_sum(dist_metric * pos_targets_norm)
      losses.append(a_loss * loss_config.a)

    print_op = tf.print(f"losses {loss_config._repr}:", to_log, output_stream=sys.stderr)
    with tf.control_dependencies([print_op]):
      self.total_loss = tf.add_n(losses)

    # self.total_loss = tf.add_n(losses)

    # tf.summary.scalar("losses/ent_loss", loss)
    # self.total_loss = loss

    if self.mode == "eval":
      # Forward and backward scores
      f_scores = scores[:-1]
      b_scores = scores[1:]

      f_max = tf.to_int64(tf.argmax(f_scores, axis=1))
      b_max = tf.to_int64(tf.argmax(b_scores, axis=1))

      targets = list(range(FLAGS.batch_size - 1))
      targets = tf.constant(targets, dtype=tf.int64)
      fwd_targets = targets + 1

      names_to_values, names_to_updates = tf.contrib.slim.metrics.aggregate_metric_map({
        "Acc/Fwd Acc": tf.contrib.slim.metrics.streaming_accuracy(f_max, fwd_targets),
        "Acc/Bwd Acc": tf.contrib.slim.metrics.streaming_accuracy(b_max, targets)
      })

      for name, value in names_to_values.items():
        tf.summary.scalar(name, value)

      self.eval_op = list(names_to_updates.values())

  def build(self):
    """Creates all ops for training, evaluation or encoding."""
    self.build_inputs()
    self.build_word_embeddings()
    self.build_encoder()
    self.build_loss()

  def build_enc(self):
    """Creates all ops for training, evaluation or encoding."""
    self.build_inputs()
    self.build_word_embeddings()
    self.build_encoder()
