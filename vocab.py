from collections import Counter
import pickle

from gensim.matutils import corpus2csc
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


class VocabDict(object):
    def __init__(self, special_tokens=("<PAD>", "<GO>", "<EOS>", "<UNK>")):
        self.special_tokens = special_tokens
        self.initialise_vocab()
        self.update_id2token()
        self.freqs = {"df": Counter(), "tf": Counter()}

    def initialise_vocab(self):
        self.token2id = {token: i for i, token in enumerate(self.special_tokens)}

    def update_id2token(self):
        self.id2token = {self.token2id[token]: token for token in self.token2id}

    def __len__(self):
        return len(self.token2id)

    def get(self, token, update=False):
        if token in self.token2id:
            return self.token2id[token]
        elif update:
            self.token2id[token] = len(self.token2id)
            return self.token2id[token]
        else:
            return len(self.special_tokens) - 1

    def add_documents(self, docs):
        i = 0  # in case len(docs) == 0
        for i, doc in enumerate(docs):
            if i % 10000 == 0:
                logger.debug("added %s documents. current vocab size: %s.", i, len(self))
            term_freqs = Counter(doc)
            # don't count special tokens
            term_freqs = {token: term_freqs[token] for token in term_freqs
                          if self.get(token, update=True) >= len(self.special_tokens)}
            self.freqs["tf"].update(term_freqs)
            self.freqs["df"].update({token: 1 for token in term_freqs})
        logger.info("added %s documents. current vocab size: %s.", i, len(self))
        return self

    def trim(self, max_vocab=0, min_count=2, freq_metric="df"):
        self.initialise_vocab()
        sorted_freqs = self.freqs[freq_metric].most_common(max_vocab or None)
        for token, freq in sorted_freqs:
            if min_count > 0 and freq < min_count:
                break
            self.get(token, update=True)

        self.update_id2token()
        for metric in self.freqs:
            self.freqs[metric] = Counter({token: self.freqs[metric][token]
                                          for token in self.freqs[metric]
                                          if token in self.token2id})
        logger.info("trimmed vocab size: %s.", len(self))
        return self

    def fit(self, docs, max_vocab=0, min_count=2, freq_metric="df"):
        self.add_documents(docs)
        self.trim(max_vocab, min_count, freq_metric)
        self.trim(freq_metric="tf")
        return self

    def transform(self, docs, pad_length=0):
        for doc in docs:
            if pad_length == 0:
                yield np.array([self.get(token) for token in doc], int)
            else:
                word_ids = np.zeros(pad_length, int)
                for i, token in enumerate(doc):
                    if i >= pad_length:
                        break
                    word_ids[i] = self.get(token)
                yield word_ids

    def inverse_transform(self, vecs):
        for vec in vecs:
            tokens = []
            for token_id in vec:
                if token_id == 0:
                    break
                else:
                    tokens.append(self.id2token[token_id])
            yield tokens

    @staticmethod
    def to_gensim_corpus(vecs):
        for vec in vecs:
            count = Counter(vec)
            yield count.items()

    def to_dtm(self, vecs):
        n_docs = len(vecs)
        corpus = self.to_gensim_corpus(vecs)
        dtm = corpus2csc(corpus, num_terms=len(self), dtype=int, num_docs=n_docs).T
        return dtm

    def save(self, filename):
        with open(filename, "wb") as f:
            f.write(pickle.dumps(self))
        logger.info("vocab saved: %s.", filename)

    @staticmethod
    def load(filename):
        logger.info("loading vocab: %s.", filename)
        with open(filename, "rb") as f:
            return pickle.loads(f.read())
