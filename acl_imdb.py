from argparse import ArgumentParser
import glob
import os
import sys

import pandas as pd
from bs4 import BeautifulSoup
import shutil
import tarfile
from urllib.parse import urlsplit

import requests

from logger import get_logger
from utils import clean_tokenise_text, make_dirs, PROJECT_DIR
from vocab import VocabDict

logger = get_logger(__name__)


def download_data(url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                  dest_dir="data"):
    # prepare filename
    _, _, url_path, _, _ = urlsplit(url)
    filename = os.path.basename(url_path)
    dest = os.path.join(dest_dir, filename)
    make_dirs(dest)

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
    df = clean_tokenise_text(df, min_token_len=min_token_len)
    return df


def read_data(src_dir="data", min_token_len=3):
    df = pd.read_csv(os.path.join(src_dir, "aclImdb", "raw.csv"))
    logger.info("acl imdb dataframe shape: %s.", df.shape)

    # sentiment
    df.loc[df["label"] == "pos", "sentiment"] = 1
    df.loc[df["label"] == "neg", "sentiment"] = 0

    # rating
    labeled_mask = ~df["sentiment"].isnull()
    df.loc[labeled_mask, "rating"] = (df.loc[labeled_mask, "id"]
                                      .str.split("_")
                                      .str.get(1).astype(int))

    df = (parse_reviews(df, min_token_len=min_token_len)
          .drop(["review"], axis=1))
    logger.info("reviews parsed.")
    return df


def split_df(df):
    # masks
    train_mask = df["group"] == "train"
    unlabeled_mask = df["label"] == "unsup"

    # split dataframe
    train_df = df.loc[train_mask & ~unlabeled_mask, :].reset_index(drop=True)
    unlabeled_df = df.loc[unlabeled_mask, :].reset_index(drop=True)
    test_df = df.loc[~train_mask, :].reset_index(drop=True)

    return {"train": train_df, "unlabeled": unlabeled_df, "test": test_df}


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
        (outputs[key].drop(["tokens", "token_count", "token_ids"], axis=1)
         .to_csv(os.path.join(src_dir, key + ".csv"), index=False))
    vocab.save(os.path.join(src_dir, "vocab.pkl"))
    logger.info("prepared data saved: %s.", src_dir)

    outputs.update({"vocab": vocab})
    return outputs


def read_prepared_data(src_dir="data"):
    # load data
    src_dir = os.path.join(src_dir, "aclImdb")
    train_df = pd.read_csv(os.path.join(src_dir, "train.csv"))
    unlabeled_df = pd.read_csv(os.path.join(src_dir, "unlabeled.csv"))
    test_df = pd.read_csv(os.path.join(src_dir, "test.csv"))
    vocab = VocabDict.load(os.path.join(src_dir, "vocab.pkl"))
    logger.info("prepared data loaded: %s.", src_dir)

    # process data
    outputs = {"train": train_df, "unlabeled": unlabeled_df, "test": test_df}
    for key in outputs:
        df = outputs[key].copy()
        df["tokens"] = df["cleaned_text"].str.split(" ")
        df["token_count"] = df["tokens"].str.len()
        df["token_ids"] = vocab.transform(df["tokens"])
        outputs[key] = df
        logger.debug("%s dataframe shape: %s.", key, df.shape)

    outputs.update({"vocab": vocab})
    return outputs


if __name__ == "__main__":
    parser = ArgumentParser(description="Download, extract and prepare ACL IMDB data.")
    parser.add_argument("--url", default="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                        help="url of ACL IMDB data (default: %(default)s)")
    parser.add_argument("--dest", default="data",
                        help="destination directory of downlaoded and extracted files (default: %(default)s)")
    parser.add_argument("--log-path", default=os.path.join(PROJECT_DIR, "main.log"),
                        help="path of log file (default: %(default)s)")
    args = parser.parse_args()

    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParser: %s.", args)

    try:
        download_data(args.url, args.dest)
        parse_data(args.dest)
        prepare_data(args.dest)

    except Exception as e:
        logger.exception(e)
        raise e
