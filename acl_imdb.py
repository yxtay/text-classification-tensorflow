from argparse import ArgumentParser
import glob
import os
import sys

import pandas as pd
from bs4 import BeautifulSoup
import shutil
import tarfile
import tensorflow as tf
from urllib.parse import urlsplit

import requests

from logger import get_logger, get_name
from utils import tokenise_text, make_dirs, PROJECT_DIR
from vocab import VocabDict

name = get_name(__name__, __file__)
logger = get_logger(name)


def download_data(url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                  dest_dir="data"):
    # prepare filename
    _, _, url_path, _, _ = urlsplit(url)
    filename = os.path.basename(url_path)
    dest = os.path.join(dest_dir, filename)
    make_dirs(dest_dir)

    # downlaod tar.gz
    if not os.path.exists(dest):
        logger.info("downloading file: %s.", url)
        r = requests.get(url, stream=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        logger.info("file downloaded: %s.", dest)

    # extract tag.gz
    if not os.path.exists(os.path.join(dest_dir, "aclImdb", "README")):
        tar = tarfile.open(dest, "r:gz")
        tar.extractall(dest_dir)
        tar.close()
        logger.info("file extracted.")


def parse_data(src_dir="data"):
    src_dir = os.path.join(src_dir, "aclImdb")
    txt_files = glob.glob(os.path.join(src_dir, "*/*/*.txt"))

    # read files
    logger.info("reading %s files.", len(txt_files))
    data = []
    for fname in txt_files:
        group, label, file_id = fname.split(os.sep)[-3:]
        file_id, _ = os.path.splitext(file_id)
        with open(fname) as f:
            text = f.read()
        data.append((file_id, group, label, text))

    # create dataframe
    df = pd.DataFrame(data, columns=["id", "group", "label", "review"])
    logger.info("dataframe shape: %s.", df.shape)

    # save dataframe
    dest_csv = os.path.join(src_dir, "raw.csv")
    df.to_csv(dest_csv, index=False)
    logger.info("dataframe saved: %s.", dest_csv)
    return df


def parse_reviews(df, review_col="review", min_token_len=3):
    df = df.copy()
    df["text"] = (df[review_col].fillna("")
                  .apply(lambda s: BeautifulSoup(s, "lxml").get_text()))
    df = tokenise_text(df, min_token_len=min_token_len)
    return df


def read_data(src_dir="data", min_token_len=3):
    df = pd.read_csv(os.path.join(src_dir, "aclImdb", "raw.csv"))
    logger.info("acl imdb dataframe shape: %s.", df.shape)

    # sentiment
    df.loc[df["label"] == "pos", "sentiment"] = 1
    df.loc[df["label"] == "neg", "sentiment"] = 0

    # rating
    labelled_mask = ~df["sentiment"].isnull()
    df.loc[labelled_mask, "rating"] = (df.loc[labelled_mask, "id"]
                                       .str.split("_")
                                       .str.get(1).astype(int))

    df = (parse_reviews(df, min_token_len=min_token_len)
          .drop(["review"], axis=1))
    logger.info("reviews parsed.")
    return df


def split_df(df):
    # masks
    train_mask = df["group"] == "train"
    unlabelled_mask = df["label"] == "unsup"

    # split dataframe
    train_df = df.loc[train_mask & ~unlabelled_mask, :].reset_index(drop=True)
    unlabelled_df = df.loc[unlabelled_mask, :].reset_index(drop=True)
    test_df = df.loc[~train_mask, :].reset_index(drop=True)

    return {"train": train_df, "unlabelled": unlabelled_df, "test": test_df}


def prepare_data(src_dir="data", min_token_len=3, min_count=5):
    df = read_data(src_dir, min_token_len=min_token_len)
    src_dir = os.path.join(src_dir, "aclImdb")

    # vocab and token_ids
    train_mask = df["group"] == "train"
    vocab = VocabDict()
    vocab.fit(df.loc[train_mask, "tokens"], min_count=min_count)
    df["token_ids"] = list(vocab.transform(df["tokens"]))

    # split dataframe
    outputs = split_df(df)

    # save prepared data
    for key in outputs:
        # csv
        (outputs[key].drop(["tokens", "token_count", "token_ids"], axis=1)
         .to_csv(os.path.join(src_dir, key + ".csv"), index=False))
        # tfrecords
        to_tfrecord(outputs[key], os.path.join(src_dir, key + ".tfrecord"))
    # vocab
    vocab.save(os.path.join(src_dir, "vocab.pkl"))
    logger.info("prepared data saved: %s.", src_dir)

    outputs.update({"vocab": vocab})
    return outputs


def to_tfrecord(df, file_path):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))

    def _int64_feature_list(values):
        return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

    def _bytes_feature_list(values):
        return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

    with tf.python_io.TFRecordWriter(file_path) as writer:
        for row in df.itertuples():
            example = tf.train.SequenceExample(
                context=tf.train.Features(
                    feature={
                        "id": _bytes_feature(getattr(row, "id")),
                        "group": _bytes_feature(getattr(row, "group")),
                        "label": _bytes_feature(getattr(row, "label")),
                        "sentiment": _float_feature(getattr(row, "sentiment")),
                        "rating": _float_feature(getattr(row, "rating")),
                        "text": _bytes_feature(getattr(row, "text")),
                        "cleaned_text": _bytes_feature(getattr(row, "cleaned_text")),
                        "token_count": _int64_feature(getattr(row, "token_count")),
                    }),
                feature_lists=tf.train.FeatureLists(
                    feature_list={
                        "tokens": _bytes_feature_list(getattr(row, "tokens")),
                        "token_ids": _int64_feature_list(getattr(row, "token_ids")),
                    }))
            writer.write(example.SerializeToString())


def read_prepared_data(src_dir="data"):
    # load data
    src_dir = os.path.join(src_dir, "aclImdb")
    outputs = {key: pd.read_csv(os.path.join(src_dir, key + ".csv"))
               for key in ["train", "unlabelled", "test"]}
    vocab = VocabDict.load(os.path.join(src_dir, "vocab.pkl"))
    logger.info("prepared data loaded: %s.", src_dir)

    # process data
    for key in outputs:
        df = outputs[key].copy()
        df["tokens"] = df["cleaned_text"].str.split(" ")
        df["token_count"] = df["tokens"].str.len()
        df["token_ids"] = vocab.transform(df["tokens"])
        outputs[key] = df
        logger.debug("%s dataframe shape: %s.", key, df.shape)

    outputs.update({"vocab": vocab})
    return outputs


def tf_dataset_pipeline(features=("sentiment", "token_ids"), shuffle=True, batch_size=32):
    context_features = {
        "id": tf.FixedLenFeature((), tf.string),
        "group": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.string),
        "sentiment": tf.FixedLenFeature((), tf.float32),
        "rating": tf.FixedLenFeature((), tf.float32),
        "text": tf.FixedLenFeature((), tf.string),
        "cleaned_text": tf.FixedLenFeature((), tf.string),
        "token_count": tf.FixedLenFeature((), tf.int64),
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature((), tf.string),
        "token_ids": tf.FixedLenSequenceFeature((), tf.int64),
    }

    def parse_tfrecord(serialized_example):
        parsed_context, parsed_sequence = tf.parse_single_sequence_example(
            serialized_example,
            context_features={key: context_features[key]
                              for key in context_features
                              if key in features},
            sequence_features={key: sequence_features[key]
                               for key in sequence_features
                               if key in features}
        )
        parsed_features = parsed_context.copy()
        parsed_features.update(parsed_sequence)
        return parsed_features

    file_input = tf.placeholder(tf.string, [None], "file_inputs")

    dataset = tf.data.TFRecordDataset(file_input)
    dataset = dataset.map(parse_tfrecord)
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes={key: [None] if key in sequence_features else []
                                                  for key in features})

    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()
    return {"file_input": file_input, "dataset": dataset,
            "data_iterator": iterator, "data_element": element}


def read_tfrecord(src_dir="data"):
    # load data
    src_dir = os.path.join(src_dir, "aclImdb")
    file_input = [os.path.join(src_dir, "train.tfrecord")]

    dataset_pipeline = tf_dataset_pipeline(["id", "sentiment", "rating", "tokens", "token_ids"])
    with tf.Session() as sess:
        sess.run(dataset_pipeline["data_iterator"].initializer,
                 {dataset_pipeline["file_input"]: file_input})
        print(sess.run(dataset_pipeline["data_element"]))


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare ACL IMDB data.")
    parser.add_argument("--url", default="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                        help="url of ACL IMDB data (default: %(default)s)")
    parser.add_argument("--dest", default="data",
                        help="destination directory of downlaoded and extracted files (default: %(default)s)")
    parser.add_argument("--log-path", default=os.path.join(PROJECT_DIR, "main.log"),
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(name, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        download_data(args.url, args.dest)
        parse_data(args.dest)
        prepare_data(args.dest)

    except Exception as e:
        logger.exception(e)
        raise e
