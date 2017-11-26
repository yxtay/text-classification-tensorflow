import os
import re

import pandas as pd
import tensorflow as tf

###
# file system
###

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


def make_dirs(path, empty=False):
    """
    create dir and clear if required
    """
    os.makedirs(path, exist_ok=True)

    if empty:
        files = [os.path.join(path, item) for item in os.listdir(path)]
        for item in files:
            if os.path.isfile(item):
                os.remove(item)
    return path


def path_join(*paths, empty=False):
    """
    join paths and create dir
    """
    path = os.path.abspath(os.path.join(*paths))
    make_dirs(os.path.dirname(path), empty)
    return path


###
# data processing
##

def tokenise_text(df, text_col="text", token_regex=r"[A-Za-z]+", min_token_len=3):
    df = df.copy()
    df["tokens"] = (df[text_col].fillna("")
                    .str.lower()
                    .str.findall(token_regex, flags=re.IGNORECASE)
                    .map(lambda doc: [token for token in doc
                                      if len(token) >= min_token_len]))
    df["cleaned_text"] = df["tokens"].str.join(" ")
    df["token_count"] = df["tokens"].str.len()
    return df


def pad_boundary_tokens(df, tokens_col="tokens", go="<GO>", eos="<EOS>"):
    n = len(df[tokens_col])
    go_series = pd.Series([go] * n)
    eos_series = pd.Series([eos] * n)
    df[tokens_col] = (go_series
                      .str.cat(df[tokens_col].str.join(" "), sep=" ")
                      .str.cat(eos_series, sep=" ")
                      .str.split())
    return df


###
# tf functions
###

def length(sequence):
    length = tf.reduce_sum(tf.sign(sequence), 1)
    length = tf.cast(length, tf.int32)
    return length


def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)
