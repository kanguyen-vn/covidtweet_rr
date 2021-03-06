from absl import flags

import os
import numpy as np
import six
import tensorflow as tf
import tensorflow_ranking as tfr

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

flags.DEFINE_string("train_path", os.path.join(root_dir, "data", "model_input", "input_train.txt"),
                    "Input file path used for training.")
flags.DEFINE_string("vali_path", os.path.join(root_dir, "data", "model_input", "input_val.txt"),
                    "Input file path used for validation.")
flags.DEFINE_string("test_path", os.path.join(root_dir, "data", "model_input", "input_test.txt"),
                    "Input file path used for testing.")
flags.DEFINE_string("output_dir", os.path.join(current_dir, "tf_ranking_logs"),
                    "Output directory for models.")

flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 100000,
                     "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.005, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5,
                   "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["64", "32"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("num_features", 100, "Number of features per document.")
flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for the primary loss function.")
flags.DEFINE_string(
    "secondary_loss", None, "The RankingLossKey for the secondary loss for "
    "multi-objective learning.")
flags.DEFINE_float(
    "secondary_loss_weight", 0.5, "The weight for the secondary loss in "
    "multi-objective learning.")

FLAGS = flags.FLAGS

_PRIMARY_HEAD = "primary_head"
_SECONDARY_HEAD = "secondary_head"


def _use_multi_head():
    """Returns True if using multi-head."""
    return FLAGS.secondary_loss is not None


class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


def example_feature_columns():
    """Returns the example feature columns."""
    feature_names = ["{}".format(i + 1) for i in range(FLAGS.num_features)]
    return {
        name:
        tf.feature_column.numeric_column(name, shape=(1,), default_value=0.0)
        for name in feature_names
    }


def load_libsvm_data(path, list_size):
    """Returns features and labels in numpy.array."""

    def _parse_line(line):
        """Parses a single line in LibSVM format."""
        tokens = line.split("#")[0].split()
        assert len(tokens) >= 2, "Ill-formatted line: {}".format(line)
        label = float(tokens[0])
        qid = tokens[1]
        kv_pairs = [kv.split(":") for kv in tokens[2:]]
        features = {k: float(v) for (k, v) in kv_pairs}
        return qid, features, label

    tf.compat.v1.logging.info("Loading data from {}".format(path))

    # The 0-based index assigned to a query.
    qid_to_index = {}
    # The number of docs seen so far for a query.
    qid_to_ndoc = {}
    # Each feature is mapped an array with [num_queries, list_size, 1]. Label has
    # a shape of [num_queries, list_size]. We use list for each of them due to the
    # unknown number of queries.
    feature_map = {k: [] for k in example_feature_columns()}
    label_list = []
    total_docs = 0
    discarded_docs = 0
    with open(path, "rt") as f:
        for line in f:
            qid, features, label = _parse_line(line)
            if qid not in qid_to_index:
                # Create index and allocate space for a new query.
                qid_to_index[qid] = len(qid_to_index)
                qid_to_ndoc[qid] = 0
                for k in feature_map:
                    feature_map[k].append(
                        np.zeros([list_size, 1], dtype=np.float32))
                label_list.append(np.ones([list_size], dtype=np.float32) * -1.)
            total_docs += 1
            batch_idx = qid_to_index[qid]
            doc_idx = qid_to_ndoc[qid]
            qid_to_ndoc[qid] += 1
            # Keep the first 'list_size' docs only.
            if doc_idx >= list_size:
                discarded_docs += 1
                continue
            for k, v in six.iteritems(features):
                assert k in feature_map, "Key {} not found in features.".format(
                    k)
                feature_map[k][batch_idx][doc_idx, 0] = v
            label_list[batch_idx][doc_idx] = label

    tf.compat.v1.logging.info(
        "Number of queries: {}".format(len(qid_to_index)))
    tf.compat.v1.logging.info(
        "Number of documents in total: {}".format(total_docs))
    tf.compat.v1.logging.info(
        "Number of documents discarded: {}".format(discarded_docs))

    # Convert everything to np.array.
    for k in feature_map:
        feature_map[k] = np.array(feature_map[k])
    return feature_map, np.array(label_list)


def get_train_inputs(features, labels, batch_size):
    """Set up training input in batches."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _train_input_fn():
        """Defines training input fn."""
        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }
        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(
                labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices(
            (features_placeholder, labels_placeholder))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _train_input_fn, iterator_initializer_hook


def get_eval_inputs(features, labels):
    """Set up eval inputs in a single batch."""
    iterator_initializer_hook = IteratorInitializerHook()

    def _eval_input_fn():
        """Defines eval input fn."""
        features_placeholder = {
            k: tf.compat.v1.placeholder(v.dtype, v.shape)
            for k, v in six.iteritems(features)
        }
        if _use_multi_head():
            placeholder = tf.compat.v1.placeholder(labels.dtype, labels.shape)
            labels_placeholder = {
                _PRIMARY_HEAD: placeholder,
                _SECONDARY_HEAD: placeholder,
            }
        else:
            labels_placeholder = tf.compat.v1.placeholder(
                labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensors(
            (features_placeholder, labels_placeholder))
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        if _use_multi_head():
            feed_dict = {
                labels_placeholder[head_name]: labels
                for head_name in labels_placeholder
            }
        else:
            feed_dict = {labels_placeholder: labels}
        feed_dict.update(
            {features_placeholder[k]: features[k] for k in features_placeholder})
        iterator_initializer_hook.iterator_initializer_fn = (
            lambda sess: sess.run(iterator.initializer, feed_dict=feed_dict))
        return iterator.get_next()

    return _eval_input_fn, iterator_initializer_hook


def make_serving_input_fn():
    """Returns serving input fn to receive tf.Example."""
    feature_spec = tf.feature_column.make_parse_example_spec(
        example_feature_columns().values())
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec)


def make_transform_fn():
    """Returns a transform_fn that converts features to dense Tensors."""

    def _transform_fn(features, mode):
        """Defines transform_fn."""
        if mode == tf.estimator.ModeKeys.PREDICT:
            # We expect tf.Example as input during serving. In this case, group_size
            # must be set to 1.
            if FLAGS.group_size != 1:
                raise ValueError(
                    "group_size should be 1 to be able to export model, but get %s" %
                    FLAGS.group_size)
            context_features, example_features = (
                tfr.feature.encode_pointwise_features(
                    features=features,
                    context_feature_columns=None,
                    example_feature_columns=example_feature_columns(),
                    mode=mode,
                    scope="transform_layer"))
        else:
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=None,
                example_feature_columns=example_feature_columns(),
                mode=mode,
                scope="transform_layer")

        return context_features, example_features

    return _transform_fn


def make_score_fn():
    """Returns a groupwise score fn to build `EstimatorSpec`."""

    def _score_fn(unused_context_features, group_features, mode, unused_params,
                  unused_config):
        """Defines the network to score a group of documents."""
        with tf.compat.v1.name_scope("input_layer"):
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(example_feature_columns())
            ]
            input_layer = tf.concat(group_input, 1)
            tf.compat.v1.summary.scalar("input_sparsity",
                                        tf.nn.zero_fraction(input_layer))
            tf.compat.v1.summary.scalar("input_max",
                                        tf.reduce_max(input_tensor=input_layer))
            tf.compat.v1.summary.scalar("input_min",
                                        tf.reduce_min(input_tensor=input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.compat.v1.layers.batch_normalization(
            input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.compat.v1.summary.scalar("fully_connected_{}_sparsity".format(i),
                                        tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.compat.v1.layers.dropout(
            cur_layer, rate=FLAGS.dropout_rate, training=is_training)
        logits = tf.compat.v1.layers.dense(cur_layer, units=FLAGS.group_size)
        if _use_multi_head():
            # Duplicate the logits for both heads.
            return {_PRIMARY_HEAD: logits, _SECONDARY_HEAD: logits}
        else:
            return logits

    return _score_fn


def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    metric_fns.update({
        "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
            # tfr.metrics.RankingMetricKey.ARP,
            # tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
            tfr.metrics.RankingMetricKey.MRR,
            tfr.metrics.RankingMetricKey.MAP,
            # tfr.metrics.RankingMetricKey.PRECISION,
            # tfr.metrics.RankingMetricKey.RECALL,
        ]
    })
    metric_fns.update({
        # "metric/ndcg@%d" % topn: tfr.metrics.make_ranking_metric_fn(
        #     tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        # for topn in [1, 3, 5, 10]
        "metric/%s@%d" % (name, topn): tfr.metrics.make_ranking_metric_fn(
            name, topn=topn)
        for name in [
            # tfr.metrics.RankingMetricKey.MRR,
            # tfr.metrics.RankingMetricKey.MAP,
            tfr.metrics.RankingMetricKey.PRECISION,
            tfr.metrics.RankingMetricKey.RECALL,
        ]
        for topn in [1, 3, 5, 10]
    })
    return metric_fns


def train_and_eval():
    """Train and Evaluate."""

    features, labels = load_libsvm_data(FLAGS.train_path, FLAGS.list_size)
    train_input_fn, train_hook = get_train_inputs(features, labels,
                                                  FLAGS.train_batch_size)

    features_vali, labels_vali = load_libsvm_data(FLAGS.vali_path,
                                                  FLAGS.list_size)
    vali_input_fn, vali_hook = get_eval_inputs(features_vali, labels_vali)

    features_test, labels_test = load_libsvm_data(FLAGS.test_path,
                                                  FLAGS.list_size)
    test_input_fn, test_hook = get_eval_inputs(features_test, labels_test)

    optimizer = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=FLAGS.learning_rate)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        update_ops = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step())
        train_op = tf.group([minimize_op, update_ops])
        return train_op

    if _use_multi_head():
        primary_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_PRIMARY_HEAD)
        secondary_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.secondary_loss),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn,
            name=_SECONDARY_HEAD)
        ranking_head = tfr.head.create_multi_ranking_head(
            [primary_head, secondary_head], [1.0, FLAGS.secondary_loss_weight])
    else:
        ranking_head = tfr.head.create_ranking_head(
            loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
            eval_metric_fns=get_eval_metric_fns(),
            train_op_fn=_train_op_fn)

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=FLAGS.group_size,
            transform_fn=make_transform_fn(),
            ranking_head=ranking_head),
        config=tf.estimator.RunConfig(
            FLAGS.output_dir, save_checkpoints_steps=1000))

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        hooks=[train_hook],
        max_steps=FLAGS.num_train_steps)
    # Export model to accept tf.Example when group_size = 1.
    if FLAGS.group_size == 1:
        vali_spec = tf.estimator.EvalSpec(
            input_fn=vali_input_fn,
            hooks=[vali_hook],
            steps=1,
            exporters=tf.estimator.LatestExporter(
                "latest_exporter",
                serving_input_receiver_fn=make_serving_input_fn()),
            start_delay_secs=0,
            throttle_secs=30)
    else:
        vali_spec = tf.estimator.EvalSpec(
            input_fn=vali_input_fn,
            hooks=[vali_hook],
            steps=1,
            start_delay_secs=0,
            throttle_secs=30)

    # Train and validate
    tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

    # Evaluate on the test data.
    estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])
    return (estimator, train_spec, vali_spec)


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    train_and_eval()


if __name__ == "__main__":
    # flags.mark_flag_as_required("train_path")
    # flags.mark_flag_as_required("vali_path")
    # flags.mark_flag_as_required("test_path")
    # flags.mark_flag_as_required("output_dir")

    tf.compat.v1.app.run()
