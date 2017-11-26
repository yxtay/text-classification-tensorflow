import tensorflow as tf
from tensorflow.contrib import layers

from logger import get_logger, get_name

name = get_name(__name__, __file__)
logger = get_logger(name)


def build_infer_graph(features, params):
    """
    builds inference graph
    """
    _params = {
        "batch_size": 32,
        "p_keep": 1.0,
        "vocab_size": None,
        "hidden_size": 32,
    }
    _params.update(params)
    logger.debug("building inference graph: %s.", _params)

    # other placeholders
    batch_size = tf.placeholder_with_default(_params["batch_size"], [], "batch_size")
    p_keep = tf.placeholder_with_default(_params["p_keep"], [], "p_keep")

    # embedding layer
    with tf.variable_scope("embedding"):
        embedding_weights = tf.get_variable("weights", [params["vocab_size"], params["hidden_size"]],
                                            initializer=tf.glorot_normal_initializer())
        embed_seq = tf.nn.embedding_lookup(embedding_weights, features["token_ids"])
        # shape: [batch_size, seq_len, hidden_size]
        # remove padding weights
        embed_seq = embed_seq * tf.expand_dims(tf.sign(features["token_ids"]), axis=2)
        # shape: [batch_size, seq_len, hidden_size]
        embed_seq = tf.nn.dropout(embed_seq, keep_prob=p_keep)
        # shape: [batch_size, seq_len, hidden_size]

    with tf.variable_scope("aggregate_embedding"):
        embed_agg = tf.reduce_sum(embed_seq, axis=1)
        # shape: [batch_size, hidden_size]
        embed_agg /= tf.reduce_sum(tf.sign(features["token_ids"]), axis=1)
        # shape: [batch_size, hidden_size]

    with tf.variable_scope("fully_connected"):
        dense_weights = tf.get_variable("weights", shape=[params["hidden_size"], 1])
        dense_bias = tf.get_variable("bias", shape=[1])
        logits = tf.nn.xw_plus_b(embed_agg, dense_weights, dense_bias)
        # shape: [batch_size, 1]
        probs = tf.nn.sigmoid(logits)
        # shape: [batch_size, 1]

        tf.summary.histogram("logits", logits)
        tf.summary.histogram("probs", probs)

    with tf.name_scope("embedding"):
        tf.summary.histogram("sequence", embed_seq)
        tf.summary.histogram("document", embed_agg)

    model = {"logits": logits, "probs": probs,
             "p_keep": p_keep, "batch_size": batch_size, "infer_args": _params}
    return model


def build_eval_graph(features, model):
    """
    builds evaluation graph
    """
    _params = {}
    logger.debug("building evaluation graph: %s.", _params)

    with tf.variable_scope("loss"):
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.expand_dims(features["sentiment"], axis=1),
                                               logits=model["logits"])

    model = {"loss": loss, "eval_args": _params}
    return model


def build_train_graph(model, params):
    """
    builds training graph
    """
    _params = {
        "learning_rate": 0.001,
        "clip_norm": 5.0,
    }
    _params.update(params)

    logger.debug("building training graph: %s.", _params)

    learning_rate = tf.placeholder_with_default(_params["learning_rate"], [], "learning_rate")
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = layers.optimize_loss(model["loss"], global_step, learning_rate, "Adam",
                                    clip_gradients=_params["clip_norm"])

    model = {"global_step": global_step, "train_op": train_op,
             "learning_rate": learning_rate, "train_args": _params}
    return model


def build_graph():
    """
    builds graph end-to-end, including data placeholders and saver
    """
    model_args = {"batch_size": batch_size, "vocab_size": vocab_size,
                  "embedding_size": embedding_size, "rnn_size": rnn_size,
                  "num_layers": num_layers, "p_keep": p_keep,
                  "learning_rate": learning_rate, "clip_norm": clip_norm}
    logger.info("building model: %s.", model_args)

    # data placeholders
    x = tf.placeholder(tf.int32, [None, None], "X")
    # shape: [batch_size, seq_len]
    y = tf.placeholder(tf.int32, [None, None], "Y")
    # shape: [batch_size, seq_len]

    model = {"X": x, "Y": y, "args": model_args}
    model.update(build_infer_graph(model["X"],
                                   batch_size=batch_size,
                                   vocab_size=VOCAB_SIZE,
                                   embedding_size=embedding_size,
                                   rnn_size=rnn_size,
                                   num_layers=num_layers,
                                   p_keep=p_keep))
    if build_eval or build_train:
        model.update(build_eval_graph(model["logits"], model["Y"]))
    if build_train:
        model.update(build_train_graph(model["loss"],
                                       learning_rate=learning_rate,
                                       clip_norm=clip_norm))
    # init op
    model["init_op"] = tf.global_variables_initializer()
    # tensorboard summary
    model["summary"] = tf.summary.merge_all()
    # saver
    model["saver"] = tf.train.Saver()
    return model
