import glob
import os
import re

import pandas as pd
from bs4 import BeautifulSoup

from logger import get_logger
from vocab import VocabDict

logger = get_logger(__name__)


def parse_acl_imdb(path_pattern="data/aclImdb",
                   dest_csv="data/aclImdb/raw.csv"):
    txt_files = glob.glob(os.path.join(path_pattern, "*/*/*.txt"))

    logger.info("reading %s files.", len(txt_files))
    data = []
    for fname in txt_files:
        group, label, file_id = fname.split(os.sep)[-3:]
        file_id, _ = os.path.splitext(file_id)
        with open(fname) as f:
            text = f.read()
        data.append((file_id, group, label, text))

    df = pd.DataFrame(data, columns=["id", "group", "label", "review"])
    logger.info("dataframe shape: %s.", df.shape)

    if dest_csv:
        df.to_csv(dest_csv, index=False)
        logger.info("dataframe saved: %s.", dest_csv)
    return df


def parse_reviews(df, review_col="review", min_token_len=3):
    df = df.copy()
    df["text"] = (df[review_col].fillna("")
                  .apply(lambda s: BeautifulSoup(s, "lxml").get_text()))
    df = clean_tokenise_text(df, min_token_len=min_token_len)
    return df


def read_acl_imdb(src_csv="data/aclImdb/raw.csv", min_token_len=3):
    df = pd.read_csv(src_csv)
    logger.info("acl imdb dataframe shape: %s.", df.shape)

    # sentiment
    df.loc[df["label"] == "pos", "sentiment"] = 1
    df.loc[df["label"] == "neg", "sentiment"] = 0

    # rating
    labeled_mask = ~df["sentiment"].isnull()
    df.loc[labeled_mask, "rating"] = (df.loc[labeled_mask, "id"]
                                      .str.split("_")
                                      .str.get(1).astype(int))

    df = parse_reviews(df, min_token_len=min_token_len)
    logger.info("reviews parsed.")

    return df


def prepare_acl_imdb(src_csv="data/aclImdb/raw.csv", min_token_len=3, min_count=5):
    df = read_acl_imdb(src_csv, min_token_len=min_token_len)

    train_mask = df["group"] == "train"
    unlabeled_mask = df["sentiment"].isnull()

    # vocab and token_ids
    vocab = VocabDict()
    vocab.fit(df.loc[train_mask, "tokens"], min_count=min_count)
    df["token_ids"] = list(vocab.transform(df["tokens"]))

    # split dataframe
    cols = ["id", "sentiment", "rating", "text", "tokens", "token_ids"]
    train_df = df.loc[train_mask & ~unlabeled_mask, cols].reset_index(drop=True)
    unlabeled_df = df.loc[unlabeled_mask, cols].reset_index(drop=True)
    test_df = df.loc[~train_mask, cols].reset_index(drop=True)

    return {"train": train_df, "unlabeled": unlabeled_df, "test": test_df, "vocab": vocab}


def clean_tokenise_text(df, text_col="text", token_regex=r"[A-Za-z]+", min_token_len=3):
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
