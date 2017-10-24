from argparse import ArgumentParser
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from data_utils import prepare_acl_imdb
from logger import get_logger

logger = get_logger(__name__)


def prepare_fasttext_txt(df, text_col="cleaned_text", label_col=None, txt_out=None):
    if label_col is not None:
        n = df[label_col].shape[0]
        fasttext_series = (pd.Series(["__label__"] * n)
                           .str.cat(df[label_col].astype(str), sep="")
                           .str.cat(df[text_col].fillna(""), sep=" "))
    else:
        fasttext_series = df[text_col].fillna("")

    if txt_out is not None:
        with open(txt_out, "w") as f:
            f.write(fasttext_series.str.cat(sep="\n"))
        logger.info("FastText txt prepared: %s.", txt_out)
    return fasttext_series


def skipgram(fasttext_bin, train_txt, model_bin, **kwargs):
    command = [fasttext_bin, "skipgram", "-input", train_txt, "-output", model_bin]

    # custom params
    params = {"minCount": 5, "minCountLabel": 0, "wordNgrams": 1, "bucket": 2000000,
              "minn": 3, "maxn": 6, "t": 0.0001, "label": "__label__",
              "lr": 0.05, "lrUpdateRate": 100, "dim": 100, "ws": 5, "epoch": 5,
              "neg": 5, "loss": "ns", "thread": 12}
    params.update(kwargs)
    for key, value in params.items():
        if value is not None:
            command.extend(["-" + str(key), str(value)])

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


def print_word_vectors(fasttext_bin, model_bin, vocab_txt, vocab_vec):
    command = [fasttext_bin, "print-word-vectors", model_bin]

    # call FastText
    with open(vocab_txt, "r") as fin, open(vocab_vec, "w") as fout:
        logger.debug("calling FastText: %s.", " ".join(command))
        subprocess.run(command, stdin=fin, stdout=fout)

    # process vec file
    with open(vocab_vec, "r") as f:
        vecs = f.read().split("\n")
    n_row = len(vecs) - 1
    n_col = len(vecs[0].strip().split(" ")) - 1

    with open(vocab_vec, "w") as f:
        f.write("{} {}".format(n_row, n_col))
        f.write("\n")
        f.write("\n".join(vecs))
    logger.info("word vectors saved: %s.", vocab_vec)


def supervised(fasttext_bin, train_txt, model_bin, **kwargs):
    command = [fasttext_bin, "supervised", "-input", train_txt, "-output", model_bin]

    # custom params
    params = {"minCount": 1, "minCountLabel": 0, "wordNgrams": 1, "bucket": 2000000,
              "minn": 0, "maxn": 0, "t": 0.0001, "label": "__label__",
              "lr": 0.1, "lrUpdateRate": 100, "dim": 100, "ws": 5, "epoch": 5,
              "neg": 5, "loss": "softmax", "thread": 12}
    params.update(kwargs)
    for key, value in params.items():
        if value is not None:
            command.extend(["-" + str(key), str(value)])

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


def predict_prob(fasttext_bin, model_bin, test_txt, k=2, label="__label__True"):
    command = [fasttext_bin, "predict-prob", model_bin, test_txt, str(k)]

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    proc = subprocess.run(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)

    stdout_lines = proc.stdout.split("\n")
    prob = np.empty(len(stdout_lines))
    for i, line in enumerate(stdout_lines):
        prob_dict = dict(zip(*[iter(line.split())] * 2))
        prob[i] = float(prob_dict.get(label, 0))
    return prob


def test(fasttext_bin, model_bin, test_txt, k=1):
    command = [fasttext_bin, "test", model_bin, test_txt, str(k)]
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


if __name__ == "__main__":
    parser = ArgumentParser(description="Text classification with FastText.")
    parser.add_argument("--fasttext-bin", required=True,
                        help="path to fasttext bin")
    parser.add_argument("--model-dir", default="checkpoints/fasttext",
                        help="directory path to save model")
    parser.add_argument("--model-file", default="supervised",
                        help="file name of supervised model")
    parser.add_argument("--pretrained", action="store_true",
                        help="whether to use pretrained vectors on unlabelled text")
    parser.add_argument("--skipgram-file", default="skipgram",
                        help="file name of skipgram model")

    args = parser.parse_args()
    logger = get_logger(__name__, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParse: %s.", args)

    try:
        os.makedirs(args.model_dir, exist_ok=True)
        params = {}

        acl_imdb = prepare_acl_imdb()
        train_df = acl_imdb["train"]
        test_df = acl_imdb["test"]
        train_df["sentiment"] = train_df["sentiment"].astype(bool)
        test_df["sentiment"] = test_df["sentiment"].astype(bool)
        train_df["cleaned_text"] = train_df["tokens"].str.join(" ")
        test_df["cleaned_text"] = test_df["tokens"].str.join(" ")

        if args.pretrained:
            unlabeled_df = acl_imdb["unlabeled"]
            unlabeled_df["cleaned_text"] = unlabeled_df["tokens"].str.join(" ")
            unlabeled_df = unlabeled_df.append(train_df)

            unlabeled_txt = os.path.join(args.model_dir, args.skipgram_file + ".txt")
            prepare_fasttext_txt(unlabeled_df, text_col="cleaned_text", txt_out=unlabeled_txt)

            skipgram_bin = os.path.join(args.model_dir, args.skipgram_file)
            skipgram(args.fasttext_bin, unlabeled_txt, skipgram_bin)
            params.update({"pretrainedVectors": skipgram_bin + ".vec"})

        train_txt = os.path.join(args.model_dir, args.model_file + ".txt")
        prepare_fasttext_txt(train_df, text_col="cleaned_text", label_col="sentiment", txt_out=train_txt)
        test_txt = os.path.join(args.model_dir, "test.txt")
        prepare_fasttext_txt(test_df, text_col="cleaned_text", label_col="sentiment", txt_out=test_txt)

        model_bin = os.path.join(args.model_dir, args.model_file)
        supervised(args.fasttext_bin, train_txt, model_bin, **params)
        test(args.fasttext_bin, model_bin + ".bin", test_txt)
        probs = predict_prob(args.fasttext_bin, model_bin + ".bin", test_txt)

    except Exception as e:
        logger.exception(e)
