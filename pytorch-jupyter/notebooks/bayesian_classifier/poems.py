import math
import requests
from itertools import chain
from typing import Iterable, Any, List, Union, Tuple, Dict

import numpy as np
import pandas as pd
import dask.array as da
import dask
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, coo_matrix
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

def initialize_tmat(width):
    data = []
    row_ind = []
    col_ind = []
    return coo_matrix((data, (row_ind, col_ind)), shape=(width, width))

def tmat_update(
        sequence: Iterable[int],
        tmat: coo_matrix
    ) -> None:
    if len(sequence) < 2:
        return

    # Create arrays to store row and column indices for the updates
    rows, cols = [], []

    # Build the row and column indices based on the transitions in the sequence
    for i in range(len(sequence) - 1):
        rows.append(sequence[i])
        cols.append(sequence[i+1])

    # Create a COO matrix with all data set to ones (since we're counting transitions)
    # and the shape matching the target CSR matrix
    data = np.ones(len(rows))
    updates = coo_matrix((data, (rows, cols)), shape=tmat.shape)

    # Add the COO matrix to the original CSR matrix
    tmat += updates
    return tmat

def build_tmat(
        vec_arr: 'np.array', 
        vocab_size: int, 
    ) -> 'coo_matrix':
    print(f"Building transition matrix with size {vocab_size} X {vocab_size}")
    # Initialize matrices with zeros
    tmat = initialize_tmat(vocab_size)

    # For the initial states and transitions
    for sequence in vec_arr:
        tmat = tmat_update(sequence, tmat)

    return tmat

def normalize_tmat(
        tmat: 'coo_matrix'
    ) -> 'csr_matrix':
    row_sums = np.array(tmat.sum(axis=1)).ravel()

    # Compute the inverse of the row sums (avoiding division by zero)
    inv_row_sums = np.reciprocal(row_sums, where=row_sums!=0)

    # Use broadcasting to normalize rows
    normalized_trans_mat = tmat.multiply(inv_row_sums.reshape(-1, 1))
    normalized_trans_mat = normalized_trans_mat.tocsr()

    return normalized_trans_mat, row_sums

def encode_sequence(
        sequence: Iterable[str],
        word2idx: Dict[str, int]
    ) -> List[int]:
    return [word2idx.get(word, word2idx.get("$NEWCHAR$")) for word in sequence]

def get_transition_logprob(
        tmat: 'csr_matrix', 
        i: int, 
        j: int, 
        row_sums: List[float],
        epsilon_smoothing: float=0.5, 
    ) -> float:
    num_columns = tmat.shape[1]
    total_smoothing = epsilon_smoothing * num_columns
    
    observed_p = tmat[i, j]
    
    # Compute the probability including smoothing
    smoothed_p = (observed_p + epsilon_smoothing) / (row_sums[i] + total_smoothing)
    
    return np.log(smoothed_p)

def get_sequence_logprob(
        encoded_sequence: List[int], 
        tmat: 'csr_matrix', 
        row_sums: List[float], 
        epsilon_smoothing: float=0.1, 
    ) -> float:
    argsum = 0
    for idx in range(len(encoded_sequence)-1):
        i = encoded_sequence[idx]
        j = encoded_sequence[idx+1]
        argsum += get_transition_logprob(tmat, i, j, row_sums, epsilon_smoothing)
    return argsum

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
    def raw_tmat(self):
        if not hasattr(self, '_raw_tmat'):
            self._raw_tmat = build_tmat(self.vectorized_arr, len(self.idx2word))
        return self._raw_tmat

    @property
    def tmat(self):
        if not hasattr(self, '_tmat'):
            self._tmat, self._row_sums = normalize_tmat(self.raw_tmat)
            if not verify_probabilities(self._tmat):
                print(f"Error, transition matrix probabilities for label {self.label} sum to outside acceptable range.")
        return self._tmat

    @property
    def row_sums(self):
        if not hasattr(self, '_row_sums'):
            print("self.row_sums not set. Setting self.row_sums via calling self.tmat.")
            self.tmat
        return self._row_sums

    # @property
    # def logprob_mat(self):
    #     if not hasattr(self, '_logprob_mat'):
    #         self._logprob_mat = get_logprob_mat(self.tmat)
    #     return self._logprob_mat

    def infer_logprob(self, line):
        encoded_sequence = encode_sequence(line, self.word2idx)
        return get_sequence_logprob(encoded_sequence, self.tmat, self.row_sums) + math.log(self.prevalence)

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
