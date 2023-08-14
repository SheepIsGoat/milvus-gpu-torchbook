import math
import requests
from itertools import chain
from typing import Iterable, Any, List, Union, Tuple, Dict

import numpy as np
import pandas as pd
import dask.array as da
import dask
from sklearn.model_selection import train_test_split
import spacy

def decode_text(text: str):
    return bytes(text.decode("utf-8"), "utf-8").decode("unicode_escape")

def pad_array(
        ragged_array: Iterable[Iterable[str]]
    ) -> List[List[str]]:
    width = max((len(row) for row in ragged_array))
    right_pad = lambda row: row + [""]*(width-len(row))
    return [right_pad(row) for row in ragged_array]

def tokenize_line(
        line: Iterable[str],
        nlp: 'spacy.language.Language'=spacy.load("en_core_web_sm")
    ) -> List[str]:
    return ["$BEGINTOKEN$"] + [token.lemma_ for token in nlp(line)] + ["$ENDTOKEN$"]

def text_to_array(
        text: str, 
        chunksize: int=500, 
        pad: bool=False
    ) -> 'np.array':
    enc_dec = decode_text(text)
    ragged_array = [tokenize_line(line) for line in enc_dec.split("\n")]
    array = pad_array(ragged_array) if pad else ragged_array
    return np.array(array, dtype=str if pad else object)

def test_split_text(
        text,
        pad=False
    ) -> Tuple['np.array', 'np.array']:
    array = text_to_array(text, pad=pad)
    return train_test_split(array, random_state=1)

def flatten_array(
        array: 'np.array', 
        ragged: bool=True
    ) -> 'np.array':
    if ragged:
        iterchain = chain.from_iterable(array)
        return np.array(list(iterchain), dtype=str)
    return array.ravel()

def vectorize_arr(
        array: 'np.array', 
        word2idx: Dict[str, int], 
        ragged: bool=True
    ) -> 'np.array':
    w2i_map = lambda word: word2idx.get(word, word2idx.get("$NEWCHAR$"))
    if not ragged:
        return np.vectorize(w2i_map)(array)
    return np.array([
        np.array(
            [w2i_map(word) for word in line], 
            dtype=int
        ) for line in array
    ], dtype=object)

def get_transition_matrix(
        vec_arr: 'np.array', 
        vocab_size: int, 
        epsilon_smoothing: float=0.5, 
        ragged: bool=True
    ) -> Tuple['np.array', 'np.array']:

    # Account for unknown words
    vocab_size += 1
    
    # Initialize matrices with zeros
    trans_mat = np.zeros((vocab_size, vocab_size))
    init_mat = np.zeros(vocab_size)

    # For the initial states and transitions
    if ragged:
        for sentence in vec_arr:
            if len(sentence) == 0:
                continue
            init_mat[sentence[0]] += 1
            
            for i in range(len(sentence)-1):
                trans_mat[sentence[i], sentence[i+1]] += 1
    else:
        # For the initial states
        starts = vec_arr[:, 0]
        init_counts, _ = np.histogram(starts, bins=np.arange(vocab_size + 1))
        init_mat += init_counts
    
        # For the transitions
        y, x = vec_arr[:, :-1].ravel(), vec_arr[:, 1:].ravel()
        hist, _, _ = np.histogram2d(x, y, bins=(vocab_size, vocab_size))
        trans_mat += hist

    # Apply epsilon smoothing
    e_smooth = lambda x: x + epsilon_smoothing
    e_smooth_arr = np.vectorize(e_smooth)
    
    trans_mat = e_smooth_arr(trans_mat)
    init_mat = e_smooth_arr(init_mat)

    # Normalize
    normalized_trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
    normalized_init_mat = init_mat / init_mat.sum()
    
    return normalized_trans_mat, normalized_init_mat

def get_logprob_mat(
        t_mat: 'np.array'
    ) -> 'np.array':
    vec_log = np.vectorize(math.log)
    return vec_log(t_mat)

def encode_sequence(
        sequence: Iterable[str],
        word2idx: Dict[str, int]
    ) -> List[int]:
    return [word2idx.get(word, word2idx.get("$NEWCHAR$")) for word in sequence]

def get_logprob_val(
        encoded_sequence: List[str], 
        logprob_mat: 'np.array',
        epsilon: bool=1
    ) -> float:
    argsum = 0
    for i in range(len(encoded_sequence)-1):
        argsum += logprob_mat[encoded_sequence[i], encoded_sequence[i+1]]
    return argsum + epsilon

def verify_probabilities(
        t_mat: 'np.array', 
        max_err: float=10**-10
    ) -> bool:
    return [i for i in np.sum(t_mat, axis=1) if abs(i-1)>max_err] == []

def get_vocab(
        corpus: np.array
    ) -> 'np.array':
    base = np.array(["$NEWCHAR$"])
    unique_vocab = np.unique(flatten_array(corpus))
    return np.concatenate([unique_vocab, base])

def stream_lines(url, lines_per_chunk=100):
    buffer = []
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            decoded_line = line.decode('utf-8')
            buffer.append(decoded_line)
            if len(buffer) == lines_per_chunk:
                yield buffer
                buffer = []
        if buffer:  # handle any remaining lines
            yield buffer

class BayesianCorpus():
    def __init__(self, raw_txt, label, ragged=True, vocab=None):
        self.raw = raw_txt
        self.data = raw_txt  # this will become the main, since this will be chunk processing
        if isinstance(raw_txt, (str, bytes)):
            self.train, self.test = test_split_text(self.raw)  # will get rid of train/test split here and do in aggregation instead
        self.label = label
        self.ragged = ragged
        if vocab is not None:
            self._vocab = vocab
        self.prevalence = 1

    @property
    def vocab(self):
        if not hasattr(self, '_vocab'):
            self._vocab = get_vocab(self.train)
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        self._vocab = vocab

    @property
    def idx2word(self):
        return self.vocab

    @property
    def word2idx(self):
        if not hasattr(self, '_word2idx'):
            self._word2idx = {word:idx for idx, word in enumerate(self.idx2word)}
        return self._word2idx

    @property
    def vectorized_arr(self):
        if not hasattr(self, '_vectorized_arr'):
            self._vectorized_arr = vectorize_arr(self.train, self.word2idx, self.ragged)
        return self._vectorized_arr
    
    @property
    def transition_mat(self):
        if not hasattr(self, '_transition_mat'):
            self._transition_mat, self.init_prob = \
                get_transition_matrix(self.vectorized_arr, len(self.idx2word), self.ragged)
            if not verify_probabilities(self._transition_mat):
                print(f"Error, transition matrix probabilities for label {self.label} sum to outside acceptable range.")
        return self._transition_mat

    @property
    def logprob_mat(self):
        if not hasattr(self, '_logprob_mat'):
            self._logprob_mat = get_logprob_mat(self.transition_mat)
        return self._logprob_mat

    def infer_logprob(self, line):
        encoded_sequence = encode_sequence(line, self.word2idx)
        return get_logprob_val(encoded_sequence, self.logprob_mat) + math.log(self.prevalence)

def combine_vocabs(
        bayesian_corpora: Iterable['BayesianCorpus']
    ) -> None:
    vocab = np.unique(np.concatenate([bc.vocab for bc in bayesian_corpora]))
    total_num_train_docs = sum((len(bc.train) for bc in bayesian_corpora))
    for bc in bayesian_corpora:
        bc.vocab = vocab
        bc.prevalence = len(bc.train)/total_num_train_docs
    

def get_confusion_matrix(
        bayesian_corpora: 'BayesianCorpus', 
        test: bool=True
    ) -> 'pd.dataframe':
    def infer_class(doc):
        max_logprob = float('-inf')
        max_label = None
        for bc in bayesian_corpora:
            logprob = bc.infer_logprob(doc)
            if logprob > max_logprob:
                max_logprob = logprob
                max_label = bc.label
        return max_label

    classmap = {bc.label: i for i,bc in enumerate(bayesian_corpora)}
    m = len(bayesian_corpora)
    results = np.zeros((m,m), dtype=int)
    for i, b_corp in enumerate(bayesian_corpora):
        from_array = b_corp.test if test else b_corp.train
        for doc in from_array:
            j = classmap[infer_class(doc)]
            results[i, j] += 1
    raw_labels = [bc.label for bc in bayesian_corpora]
    idx_labels = ["is_" + label for label in raw_labels]
    col_labels = ["pred_" + label for label in raw_labels]
    return pd.DataFrame(results, index=idx_labels, columns=col_labels)

if __name__ == '__main__':
    import requests
    raw_poe = requests.get('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt').content
    raw_frost = requests.get('https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt').content

    frost_corpus = BayesianCorpus(raw_frost, 'frost', ragged=True)
    poe_corpus = BayesianCorpus(raw_poe, 'poe', ragged=True)

    combine_vocabs([frost_corpus, poe_corpus])

    res = get_confusion_matrix([frost_corpus, poe_corpus])
    print(res)
