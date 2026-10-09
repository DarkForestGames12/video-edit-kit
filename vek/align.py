"""Forced alignment: places each known word on the audio using a CTC acoustic model (NeMo Conformer-CTC),
Viterbi over the CTC topology. Accurate to ~40 ms, unlike recogniser timestamps (0.1-0.3 s drift)."""
import os, re
from functools import lru_cache
import numpy as np
from .common import MODELS

CTC_DIR = os.path.join(MODELS, "sherpa-onnx-nemo-ctc-en-conformer-small")
FRAME = 0.04


@lru_cache(None)
def _model():
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
    sess = ort.InferenceSession(os.path.join(CTC_DIR, "model.int8.onnx"), so, providers=["CPUExecutionProvider"])
    vocab = {}
    with open(os.path.join(CTC_DIR, "tokens.txt"), encoding="utf-8") as f:
        for line in f:
            sym, idx = line.rstrip("\n").rsplit(" ", 1)
            vocab[sym] = int(idx)
    return sess, vocab


def logprobs(x16):
    import kaldi_native_fbank as knf
    sess, vocab = _model()
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0; opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = 16000; opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, (x16 * 32768).astype(np.float32).tolist())
    fb.input_finished()
    feat = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
    feat = (feat - feat.mean(0, keepdims=True)) / (feat.std(0, keepdims=True) + 1e-5)
    f = feat.T[None].astype(np.float32)
    lp = sess.run(None, {"audio_signal": f, "length": np.array([f.shape[2]], np.int64)})[0][0]
    return lp


def tokenize(word):
    _, vocab = _model()
    w = "▁" + re.sub(r"[^a-z0-9']", "", word.lower())
    out, i = [], 0
    while i < len(w):
        for j in range(len(w), i, -1):
            if w[i:j] in vocab:
                out.append(vocab[w[i:j]]); i = j; break
        else:
            out.append(vocab.get("<unk>", 0)); i += 1
    return out


def viterbi(lp, toks):
    """CTC forced alignment. Returns the first frame of each token."""
    blank = lp.shape[1] - 1
    L = len(toks)
    S = 2 * L + 1
    lab = np.full(S, blank); lab[1::2] = toks
    T = lp.shape[0]
    NEG = -1e30
    dp = np.full(S, NEG); dp[0] = lp[0, blank]
    if S > 1: dp[1] = lp[0, lab[1]]
    bp = np.zeros((T, S), np.int8)
    skip_ok = np.zeros(S, bool)
    skip_ok[3::2] = np.array(toks[1:]) != np.array(toks[:-1]) if L > 1 else []
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate([[NEG], dp[:-1]])
        prev2 = np.where(skip_ok, np.concatenate([[NEG, NEG], dp[:-2]]), NEG)
        cand = np.stack([stay, prev1, prev2])
        arg = cand.argmax(0)
        dp = cand[arg, np.arange(S)] + lp[t, lab]
        bp[t] = arg
    s = int(S - 1 if S == 1 or dp[S - 1] >= dp[S - 2] else S - 2)
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(bp[t, s])
    first = {}
    for t, st in enumerate(path):
        if st % 2 == 1 and (st // 2) not in first:
            first[st // 2] = t
    return [first.get(k) for k in range(L)]


def align_words(x16, words, offset=0.0):
    """words: list of strings in order. Returns start times (s) for each word."""
    if not words: return []
    lp = logprobs(x16)
    toks, wid = [], []
    for i, w in enumerate(words):
        tk = tokenize(w) or [0]
        toks += tk; wid += [i] * len(tk)
    if len(toks) * 1.0 > lp.shape[0]:
        return None
    fr = viterbi(lp, toks)
    starts = [None] * len(words)
    for k, f in enumerate(fr):
        if f is not None and starts[wid[k]] is None:
            starts[wid[k]] = offset + f * FRAME
    for i in range(len(starts)):
        if starts[i] is None: starts[i] = starts[i - 1] + 0.1 if i else offset
    return starts
