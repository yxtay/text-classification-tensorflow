from argparse import ArgumentParser
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

from acl_imdb import read_prepared_data
from evaluation import get_threshold_by_cutoff
from logger import get_logger
from utils import make_dirs, PROJECT_DIR

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


def unsupervised(fasttext_bin, train_txt, model_bin, model="skipgram", **kwargs):
    command = [fasttext_bin, model, "-input", train_txt, "-output", model_bin]

    # params
    params = {"minCount": None,  # default: 5
              "minCountLabel": None,  # default: 0
              "wordNgrams": None,  # default: 1
              "bucket": None,  # default: 2000000
              "minn": None,  # default: 3
              "maxn": None,  # default: 6
              "t": None,  # default: 0.0001
              "label": None,  # default: "__label__"
              "lr": None,  # default: 0.05
              "lrUpdateRate": None,  # default: 100
              "dim": None,  # default: 100
              "ws": None,  # default: 5
              "epoch": None,  # default: 5
              "neg": None,  # default: 5
              "loss": None,  # default: "ns"
              "thread": None,  # default: 12
              "pretrainedVectors": None,  # default: None
              }
    params.update(kwargs)
    for key, value in params.items():
        if value is not None:
            command.extend(["-" + str(key), str(value)])

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


def print_word_vectors(fasttext_bin, model_bin, vocab_txt, vocab_vec):
    if not model_bin.endswith(".bin"):
        model_bin += ".bin"
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

    # params
    params = {"minCount": None,  # default: 1
              "minCountLabel": None,  # default: 0
              "wordNgrams": None,  # default: 1
              "bucket": None,  # default: 2000000
              "minn": None,  # default: 0
              "maxn": None,  # default: 0
              "t": None,  # default: 0.0001
              "label": None,  # default: "__label__"
              "lr": None,  # default: 0.1
              "lrUpdateRate": None,  # default: 100
              "dim": None,  # default: 100
              "ws": None,  # default: 5
              "epoch": None,  # default: 5
              "neg": None,  # default: 5
              "loss": None,  # default: "softmax"
              "thread": None,  # default: 12
              "pretrainedVectors": None,  # default: None
              }
    params.update(kwargs)
    for key, value in params.items():
        if value is not None:
            command.extend(["-" + str(key), str(value)])

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


def test(fasttext_bin, model_bin, test_txt, k=1):
    if not model_bin.endswith(".bin"):
        model_bin += ".bin"
    command = [fasttext_bin, "test", model_bin, test_txt, str(k)]
    logger.debug("calling FastText: %s.", " ".join(command))
    subprocess.run(command)


def predict(fasttext_bin, model_bin, test_txt, k=1, label_prefix="__label__"):
    if not model_bin.endswith(".bin"):
        model_bin += ".bin"
    command = [fasttext_bin, "predict", model_bin, test_txt, str(k)]

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    proc = subprocess.run(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)

    # process stdout
    stdout_lines = proc.stdout.split("\n")[:-1]
    pred = np.empty((len(stdout_lines), k), dtype=object)
    for i, line in enumerate(stdout_lines):
        pred[i, :] = np.array(line.replace(label_prefix, "").split(" "))
    return pred


def predict_prob(fasttext_bin, model_bin, test_txt, k=2, labels=(True,), label_prefix="__label__"):
    if not model_bin.endswith(".bin"):
        model_bin += ".bin"
    command = [fasttext_bin, "predict-prob", model_bin, test_txt, str(k)]

    # call FastText
    logger.debug("calling FastText: %s.", " ".join(command))
    proc = subprocess.run(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True)

    # process stdout
    stdout_lines = proc.stdout.split("\n")[:-1]
    prob = np.empty((len(stdout_lines), len(labels)))
    labels = [label_prefix + str(el) for el in labels]
    for i, line in enumerate(stdout_lines):
        prob_dict = dict(zip(*[iter(line.split(" "))] * 2))
        prob[i, :] = np.fromiter((prob_dict.get(el, 0) for el in labels), float)
    return prob


def print_sentence_vectors(fasttext_bin, model_bin, test_txt, test_vec):
    if not model_bin.endswith(".bin"):
        model_bin += ".bin"
    command = [fasttext_bin, "print-sentence-vectors", model_bin]

    # call FastText
    with open(test_txt, "r") as fin, open(test_vec, "w") as fout:
        logger.debug("calling FastText: %s.", " ".join(command))
        subprocess.run(command, stdin=fin, stdout=fout)


if __name__ == "__main__":
    parser = ArgumentParser(description="Text classification with FastText.")
    parser.add_argument("--fasttext-bin", required=True,
                        help="path to fasttext bin (required)")
    parser.add_argument("--data-dir", default="data",
                        help="directory path of ACL IMDB data (default: %(default)s)")
    parser.add_argument("--model-dir", default="checkpoints/fasttext",
                        help="directory path to save model (default: %(default)s)")
    parser.add_argument("--model-file", default="supervised",
                        help="file name of supervised model (default: %(default)s)")
    parser.add_argument("--pretrained", action="store_true",
                        help="whether to use pretrained vectors on unlabelled text")
    parser.add_argument("--skipgram-file", default="skipgram",
                        help="file name of skipgram model (default: %(default)s)")
    parser.add_argument("--log-path", default=os.path.join(PROJECT_DIR, "main.log"),
                        help="path of log file (default: %(default)s)")

    args = parser.parse_args()
    logger = get_logger(__name__, log_path=args.log_path, console=True)
    logger.debug("call: %s.", " ".join(sys.argv))
    logger.debug("ArgumentParse: %s.", args)

    try:
        make_dirs(args.model_dir + "/", empty=True)
        params = {}

        # load data
        acl_imdb = read_prepared_data(args.data_dir)
        train_df = acl_imdb["train"]
        test_df = acl_imdb["test"]
        train_df["sentiment"] = train_df["sentiment"].astype(bool)
        test_df["sentiment"] = test_df["sentiment"].astype(bool)
        y_true = test_df["sentiment"]

        if args.pretrained:
            # prepare training data
            unlabeled_df = acl_imdb["unlabeled"].copy().append(train_df)
            skipgram_bin = os.path.join(args.model_dir, args.skipgram_file)
            unlabeled_txt = skipgram_bin + ".txt"
            prepare_fasttext_txt(unlabeled_df, text_col="cleaned_text", txt_out=unlabeled_txt)

            # fit fasttext vectors
            logger.info("training FastText vectors.")
            unsupervised(args.fasttext_bin, unlabeled_txt, skipgram_bin)
            params.update({"pretrainedVectors": skipgram_bin + ".vec"})

        # prepare training data
        model_bin = os.path.join(args.model_dir, args.model_file)
        train_txt = model_bin + ".txt"
        prepare_fasttext_txt(train_df, text_col="cleaned_text", label_col="sentiment", txt_out=train_txt)
        test_txt = os.path.join(args.model_dir, "test.txt")
        prepare_fasttext_txt(test_df, text_col="cleaned_text", label_col="sentiment", txt_out=test_txt)

        # model training
        logger.info("training FastText model.")
        supervised(args.fasttext_bin, train_txt, model_bin, **params)
        logger.info("FastText model completed: %s.", model_bin)

        # evaluation
        test(args.fasttext_bin, model_bin, test_txt)

        y_pred = predict(args.fasttext_bin, model_bin, test_txt) == "True"
        precision, recall, f1_score, support = precision_recall_fscore_support(y_true, y_pred)
        logger.info("precision: %.4f, recall: %.4f, f1_score: %.4f, support: %s",
                    precision[1], recall[1], f1_score[1], support[1])

        y_score = predict_prob(args.fasttext_bin, model_bin, test_txt, labels=[True])
        precision, recall, threshold = get_threshold_by_cutoff(*precision_recall_curve(y_true, y_score))
        logger.info("precision: %.4f, recall: %.4f, threshold: %.4f", precision, recall, threshold)
        logger.info("roc auc: %.4f.", roc_auc_score(y_true, y_score))
        logger.info("average precision: %.4f.", average_precision_score(y_true, y_score))

        # print_sentence_vectors(args.fasttext_bin, model_bin, train_txt, model_bin + ".txt.vec")

    except Exception as e:
        logger.exception(e)
        raise e
