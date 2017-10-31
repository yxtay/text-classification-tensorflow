import os
import re

import pandas as pd

###
# file system
###

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))


def make_dirs(path, empty=False):
    """
    create dir in path and clear dir if required
    """
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    if empty:
        files = [os.path.join(dir_path, item) for item in os.listdir(dir_path)]
        for item in files:
            if os.path.isfile(item):
                os.remove(item)
    return dir_path


###
# data processing
##

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
