#!/usr/bin/env python3
"""Write a LOADABLE synthetic GGUF model: random weights, controlled shapes.

ggrun learns a machine's own speed coefficients by running its real backend on
small models that each isolate one factor (weight bytes per token, layer count,
routed experts, KV at depth, quant type). Those runs need files a llama.cpp
family backend can load and decode, which tests/build_synthetic_gguf.py (headers
only, zero-length tensors) is not. This tool is stdlib-only (no numpy, no gguf
package) so it runs on any machine that has Python 3.

    make_synthetic_gguf.py --arch dense --layers 16 --embd 512 --ff 1536 \
        --heads 8 --kv-heads 4 --quant q4_k --out synth.gguf
    make_synthetic_gguf.py --arch moe ... --experts 32 --experts-used 4 --plan

Importable: Spec, plan(spec), write_gguf(spec, fileobj), generate(spec, path).

What is written (architecture "llama" for both kinds; the MoE form is the
Mixtral-style llama architecture with expert_count/expert_used_count, so a dense
and a MoE file differ only in the feed-forward tensors):
  * the exact tensor set the llama loader creates (an extra tensor fails the load
    with "wrong number of tensors"): token_embd, per block attn_norm, attn_q/k/v/
    output, ffn_norm, then dense ffn_gate/down/up or ffn_gate_inp + ffn_*_exps,
    then output_norm and a separate output.weight; no biases, no rope_freqs;
  * a SentencePiece ("llama") vocabulary of --vocab tokens: <unk>, <s>, </s>, the
    256 <0xNN> byte tokens, "▁" and the printable ASCII characters, then filler
    pieces. Every token is unique and non-empty, as the loaders require;
  * tensor data from a deterministic pool (SHAKE-256 bytes from --seed summed into
    bell-shaped values with integer arithmetic; no libm, no random module), tiled
    with a per-tensor phase, so a multi-GiB file is written in bounded memory and
    is meant to be byte-identical on every platform and Python 3 version (CI
    compares a committed fixture; only Linux CPython 3.11 and 3.12 are checked).

Values (so a forward pass stays finite): RMSNorm gains are 1.0; every matrix has
standard deviation 1/sqrt(fan_in); the token embedding has standard deviation 1.
With pre-norm blocks each sublayer adds an O(1) update, so the residual stream
grows like sqrt(layers), never exponentially. The output head has standard
deviation logit_std/sqrt(embd) (default logit std 6), and the rows of <unk>, <s>
and </s> are exactly zero: their logit is 0 while every other token's is
N(0, 6^2), so end-of-sequence is never the greedy argmax and carries about
e^-18/vocab of the sampling mass. Callers should still send ignore_eos.

Sources mirrored here (read 2026-11-27):
  GGUF layout:  ggml-org/ggml docs/gguf.md; ggml/src/gguf.cpp (offsets must be the
                running sum of ALIGNMENT-padded sizes, name < 64 bytes, ne[0] a
                multiple of the block size)
  Block layouts: ggml-org/llama.cpp ggml/src/ggml-common.h block_q4_0, block_q8_0,
                block_q4_K; validity: ggml-quants.c ggml_validate_row_data (only the
                f16 scale fields d/dmin are checked), dequantize_row_q4_K
  Tensors/keys: llama.cpp src/models/llama.cpp (load_arch_tensors), src/llama-arch.cpp,
                src/llama-model.cpp (load_hparams), src/llama-vocab.cpp; commit
                7b74cb92d67384373e87512f4ffd2dbef756b944; ik_llama.cpp
                b945c312369effb54ec4de59bbb7f659651669e4 src/llama-load-tensors.cpp
                (create_llama_tensors), src/llama-hparams.cpp.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, asdict

GENERATOR_VERSION = 1
ALIGNMENT = 32                  # GGUF default; general.alignment is not written
GGUF_VERSION = 3
DEFAULT_VOCAB = 512
MIN_VOCAB = 3 + 256 + 95        # specials + byte tokens + "▁" and 94 printable ASCII
FLOAT_POOL = 262139             # prime: 2^18 - 5; elements of a float pool
BLOCK_POOL = 65521              # prime: 2^16 - 15; blocks of a quantised pool
EPS = 1e-5                      # llama.attention.layer_norm_rms_epsilon
ROPE_BASE = 10000.0
N_SPECIAL = 3                   # <unk>, <s>, </s>: output rows kept at zero

# gguf value types
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64 = 4, 5, 6, 7, 8, 9, 10


@dataclass(frozen=True)
class QType:
    name: str
    id: int                  # enum ggml_type
    block_bytes: int
    block_elems: int
    f16_fields: tuple        # byte offsets of f16 scale fields inside a block
    std_per_d: float         # std of a dequantised weight per unit of d (random bits)
    ftype: int               # llama_ftype for general.file_type


# std_per_d: Q4_0 w = d*(q-8), q~U{0..15}: d*sqrt(255/12); Q8_0 w = d*q, q~U{-127..127}:
# d*sqrt((255^2-1)/12); Q4_K w = d*(sc*q - 7.5*m) with dmin = 7.5*d (zero mean),
# sc,m~U{0..63}, q~U{0..15}: d*sqrt(66727.5). The test suite checks each against a
# reference dequantiser written from ggml-quants.c.
QTYPES = {
    'f32': QType('f32', 0, 4, 1, (), 1.0, 0),
    'f16': QType('f16', 1, 2, 1, (), 1.0, 1),
    'q4_0': QType('q4_0', 2, 18, 32, (0,), math.sqrt(255 / 12), 2),
    'q8_0': QType('q8_0', 8, 34, 32, (0,), math.sqrt((255 ** 2 - 1) / 12), 7),
    'q4_k': QType('q4_k', 12, 144, 256, (0, 2), math.sqrt(66727.5), 15),
}
FLOAT_TYPES = ('f32', 'f16')


@dataclass(frozen=True)
class Spec:
    arch: str = 'dense'          # dense | moe
    layers: int = 4
    embd: int = 256
    ff: int = 512                # dense FFN width; MoE: width of ONE expert
    heads: int = 4
    kv_heads: int = 0            # 0 = same as heads
    vocab: int = DEFAULT_VOCAB
    ctx: int = 65536             # context_length metadata (training context)
    experts: int = 0
    experts_used: int = 0
    quant: str = 'f16'           # type of the attention/FFN/expert matrices
    head_type: str = 'f16'       # type of token_embd and output
    seed: int = 1
    name: str = ''
    logit_std: float = 6.0

    @property
    def kv(self):
        return self.kv_heads or self.heads

    @property
    def head_dim(self):
        return self.embd // self.heads


def validate(s):
    def bad(msg):
        raise ValueError(msg)
    if s.arch not in ('dense', 'moe'):
        bad("arch must be 'dense' or 'moe'")
    for t in (s.quant, s.head_type):
        if t not in QTYPES:
            bad('unknown type %r (have %s)' % (t, ', '.join(QTYPES)))
    if s.layers < 1 or s.embd < 1 or s.ff < 1 or s.heads < 1:
        bad('layers, embd, ff and heads must be positive')
    if s.embd % s.heads:
        bad('embd %d is not a multiple of heads %d' % (s.embd, s.heads))
    if s.head_dim % 2:
        bad('head dim %d must be even (RoPE)' % s.head_dim)
    if s.heads % s.kv:
        bad('heads %d is not a multiple of kv_heads %d' % (s.heads, s.kv))
    if s.vocab < MIN_VOCAB:
        bad('vocab must be at least %d (3 specials, 256 byte tokens, 95 characters)' % MIN_VOCAB)
    if s.arch == 'moe':
        if s.experts < 2 or not 1 <= s.experts_used <= s.experts:
            bad('moe needs experts >= 2 and 1 <= experts_used <= experts')
    elif s.experts or s.experts_used:
        bad('experts and experts_used apply to --arch moe only')
    for tname, widths in ((s.quant, (('embd', s.embd), ('ff', s.ff))), (s.head_type, (('embd', s.embd),))):
        be = QTYPES[tname].block_elems
        for label, w in widths:
            if w % be:
                bad('%s %d is not a multiple of the %s block size %d' % (label, w, tname, be))
    if s.logit_std <= 0:
        bad('logit_std must be positive')


# --------------------------------------------------------------------------- vocab

def build_vocab(n):
    """(tokens, scores, token_types) of a SentencePiece vocab with n unique tokens."""
    toks = ['<unk>', '<s>', '</s>']
    types = [2, 3, 3]                                     # UNKNOWN, CONTROL, CONTROL
    scores = [0.0, 0.0, 0.0]
    for b in range(256):
        toks.append('<0x%02X>' % b)
        types.append(6)                                   # BYTE
        scores.append(0.0)
    pieces = ['▁'] + [chr(c) for c in range(33, 127)]
    for i, p in enumerate(pieces):
        toks.append(p)
        types.append(1)
        scores.append(-float(i))
    i = 0
    while len(toks) < n:
        toks.append('w%d' % i)
        types.append(1)
        scores.append(-1000.0 - i)
        i += 1
    assert len(set(toks)) == len(toks) == n
    return toks, scores, types


# --------------------------------------------------------------------------- header

def _s(x):
    b = x.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def _kv(key, vtype, payload):
    return _s(key) + struct.pack('<I', vtype) + payload


def kv_u32(k, v): return _kv(k, T_U32, struct.pack('<I', v))
def kv_u64(k, v): return _kv(k, T_U64, struct.pack('<Q', v))
def kv_f32(k, v): return _kv(k, T_F32, struct.pack('<f', v))
def kv_bool(k, v): return _kv(k, T_BOOL, struct.pack('<B', 1 if v else 0))
def kv_str(k, v): return _kv(k, T_STR, _s(v))


def kv_arr(k, etype, items, fmt=None):
    body = struct.pack('<IQ', etype, len(items))
    body += b''.join(_s(x) for x in items) if fmt is None else struct.pack('<%d%s' % (len(items), fmt), *items)
    return _kv(k, T_ARR, body)


def metadata(s):
    name = s.name or 'ggrun-synth-%s-L%d-d%d' % (s.arch, s.layers, s.embd)
    toks, scores, types = build_vocab(s.vocab)
    a = 'llama'
    kvs = [
        kv_str('general.architecture', a),
        kv_str('general.name', name),
        kv_str('general.description', 'Synthetic random-weight model for speed calibration. Not a language model.'),
        kv_u32('general.quantization_version', 2),
        kv_u32('general.file_type', QTYPES[s.quant].ftype),
        kv_u32(a + '.context_length', s.ctx),
        kv_u32(a + '.embedding_length', s.embd),
        kv_u32(a + '.block_count', s.layers),
        kv_u32(a + '.feed_forward_length', s.ff),
        kv_u32(a + '.attention.head_count', s.heads),
        kv_u32(a + '.attention.head_count_kv', s.kv),
        kv_u32(a + '.attention.key_length', s.head_dim),
        kv_u32(a + '.attention.value_length', s.head_dim),
        kv_f32(a + '.attention.layer_norm_rms_epsilon', EPS),
        kv_u32(a + '.rope.dimension_count', s.head_dim),
        kv_f32(a + '.rope.freq_base', ROPE_BASE),
        kv_u32(a + '.vocab_size', s.vocab),
    ]
    if s.arch == 'moe':
        kvs += [kv_u32(a + '.expert_count', s.experts), kv_u32(a + '.expert_used_count', s.experts_used)]
    kvs += [
        kv_str('tokenizer.ggml.model', 'llama'),
        kv_arr('tokenizer.ggml.tokens', T_STR, toks),
        kv_arr('tokenizer.ggml.scores', T_F32, scores, 'f'),
        kv_arr('tokenizer.ggml.token_type', T_I32, types, 'i'),
        kv_u32('tokenizer.ggml.bos_token_id', 1),
        kv_u32('tokenizer.ggml.eos_token_id', 2),
        kv_u32('tokenizer.ggml.unknown_token_id', 0),
        kv_bool('tokenizer.ggml.add_bos_token', True),
        kv_bool('tokenizer.ggml.add_eos_token', False),
        kv_u64('ggrun.synthetic.seed', s.seed),
        kv_u32('ggrun.synthetic.generator_version', GENERATOR_VERSION),
    ]
    return kvs


# --------------------------------------------------------------------------- plan

@dataclass
class TensorPlan:
    name: str
    dims: tuple              # ggml ne order: dims[0] is the contiguous row
    qtype: str
    role: str                # embd | head | norm | attn | ffn | router | expert
    std: float               # target std of the stored values (norms: constant 1.0)
    zero_rows: int = 0       # leading rows (dims[0] elements each) written as zero bytes
    nbytes: int = 0
    offset: int = 0          # relative to the start of the data section
    span: int = 0            # nbytes padded to ALIGNMENT (what parse_gguf.py reports)

    @property
    def elements(self):
        n = 1
        for d in self.dims:
            n *= d
        return n


def _nbytes(dims, qname):
    q = QTYPES[qname]
    rows = 1
    for d in dims[1:]:
        rows *= d
    return dims[0] // q.block_elems * q.block_bytes * rows


def tensor_list(s):
    L, d, ff, V = s.layers, s.embd, s.ff, s.vocab
    kvd = s.kv * s.head_dim
    q, h = s.quant, s.head_type
    inv = lambda fan_in: 1.0 / math.sqrt(fan_in)
    t = [TensorPlan('token_embd.weight', (d, V), h, 'embd', 1.0)]
    for i in range(L):
        p = 'blk.%d.' % i
        t += [
            TensorPlan(p + 'attn_norm.weight', (d,), 'f32', 'norm', 1.0),
            TensorPlan(p + 'attn_q.weight', (d, d), q, 'attn', inv(d)),
            TensorPlan(p + 'attn_k.weight', (d, kvd), q, 'attn', inv(d)),
            TensorPlan(p + 'attn_v.weight', (d, kvd), q, 'attn', inv(d)),
            TensorPlan(p + 'attn_output.weight', (d, d), q, 'attn', inv(d)),
            TensorPlan(p + 'ffn_norm.weight', (d,), 'f32', 'norm', 1.0),
        ]
        if s.arch == 'dense':
            t += [
                TensorPlan(p + 'ffn_gate.weight', (d, ff), q, 'ffn', inv(d)),
                TensorPlan(p + 'ffn_down.weight', (ff, d), q, 'ffn', inv(ff)),
                TensorPlan(p + 'ffn_up.weight', (d, ff), q, 'ffn', inv(d)),
            ]
        else:
            E = s.experts
            t += [
                TensorPlan(p + 'ffn_gate_inp.weight', (d, E), 'f32', 'router', inv(d)),
                TensorPlan(p + 'ffn_gate_exps.weight', (d, ff, E), q, 'expert', inv(d)),
                TensorPlan(p + 'ffn_down_exps.weight', (ff, d, E), q, 'expert', inv(ff)),
                TensorPlan(p + 'ffn_up_exps.weight', (d, ff, E), q, 'expert', inv(d)),
            ]
    t += [
        TensorPlan('output_norm.weight', (d,), 'f32', 'norm', 1.0),
        TensorPlan('output.weight', (d, V), h, 'head', s.logit_std / math.sqrt(d), zero_rows=N_SPECIAL),
    ]
    return t


@dataclass
class Plan:
    spec: Spec
    tensors: list
    header: bytes            # everything before the alignment padding
    data_start: int
    data_size: int
    file_size: int
    active_weight_bytes: int
    kv_bytes_per_token_f16: int
    warnings: list

    def to_dict(self):
        return {
            'spec': asdict(self.spec),
            'file_size': self.file_size,
            'header_bytes': len(self.header),
            'data_start': self.data_start,
            'data_size': self.data_size,
            'tensor_count': len(self.tensors),
            'active_weight_bytes': self.active_weight_bytes,
            'kv_bytes_per_token_f16': self.kv_bytes_per_token_f16,
            'type_histogram': self.type_histogram(),
            'warnings': self.warnings,
            'tensors': [{'name': t.name, 'dims': list(t.dims), 'type': t.qtype, 'nbytes': t.nbytes,
                         'offset': t.offset, 'span': t.span} for t in self.tensors],
        }

    def type_histogram(self):
        h = {}
        for t in self.tensors:
            e = h.setdefault(QTYPES[t.qtype].id, {'type': QTYPES[t.qtype].id, 'name': t.qtype,
                                                  'tensors': 0, 'elements': 0, 'bytes': 0})
            e['tensors'] += 1
            e['elements'] += t.elements
            e['bytes'] += t.span
        return [h[k] for k in sorted(h)]


def _pad(n, a=ALIGNMENT):
    return (n + a - 1) // a * a


def plan(s):
    validate(s)
    ts = tensor_list(s)
    off = 0
    for t in ts:
        if len(t.name) >= 64:
            raise ValueError('tensor name too long: ' + t.name)
        t.nbytes = _nbytes(t.dims, t.qtype)
        t.offset = off
        t.span = _pad(t.nbytes)
        off += t.span
    header = b'GGUF' + struct.pack('<IQQ', GGUF_VERSION, len(ts), len(metadata(s)))
    header += b''.join(metadata(s))
    for t in ts:
        header += _s(t.name) + struct.pack('<I', len(t.dims)) + struct.pack('<%dQ' % len(t.dims), *t.dims)
        header += struct.pack('<IQ', QTYPES[t.qtype].id, t.offset)
    data_start = _pad(len(header))
    active = 0
    for t in ts:
        if t.role == 'expert':
            active += t.nbytes * s.experts_used // s.experts
        elif t.role == 'embd':
            active += t.nbytes // s.vocab          # one row per token
        else:
            active += t.nbytes
    warn = []
    if s.head_dim not in (64, 128, 256):
        warn.append('head dim %d: flash-attention kernels commonly cover 64/128/256 only' % s.head_dim)
    return Plan(s, ts, header, data_start, off, data_start + off, active,
                2 * s.layers * s.kv * s.head_dim * 2, warn)


def format_plan(p):
    s = p.spec
    out = ['%-34s %-22s %-5s %14s %14s' % ('tensor', 'dims (ne)', 'type', 'bytes', 'offset')]
    for t in p.tensors:
        out.append('%-34s %-22s %-5s %14d %14d' % (t.name, 'x'.join(map(str, t.dims)), t.qtype, t.nbytes, t.offset))
    out.append('')
    out.append('arch=llama kind=%s layers=%d embd=%d ff=%d heads=%d kv_heads=%d vocab=%d ctx=%d%s quant=%s head=%s seed=%d' % (
        s.arch, s.layers, s.embd, s.ff, s.heads, s.kv, s.vocab, s.ctx,
        (' experts=%d used=%d' % (s.experts, s.experts_used)) if s.arch == 'moe' else '', s.quant, s.head_type, s.seed))
    out.append('tensors=%d  header=%d B  data_start=%d  data=%d B  FILE SIZE=%d B (%.3f GiB)' % (
        len(p.tensors), len(p.header), p.data_start, p.data_size, p.file_size, p.file_size / 2 ** 30))
    out.append('weight bytes read per decoded token=%d (%.3f MiB; routed experts counted k/E)' % (
        p.active_weight_bytes, p.active_weight_bytes / 2 ** 20))
    out.append('f16 KV bytes per token of context=%d' % p.kv_bytes_per_token_f16)
    out += ['warning: ' + w for w in p.warnings]
    return '\n'.join(out)


# --------------------------------------------------------------------------- data

def _seed_bytes(seed, label, n):
    return hashlib.shake_256(('ggrun-synth|%d|%s' % (seed, label)).encode()).digest(n)


class Pools:
    """Deterministic value pools, one per (type, std), tiled over each tensor."""

    def __init__(self, seed, float_len=FLOAT_POOL, block_len=BLOCK_POOL):
        self.seed, self.float_len, self.block_len = seed, float_len, block_len
        self._gauss = None
        self._cache = {}

    def _standard(self):
        """Zero-mean, unit-std values: Irwin-Hall sums of 12 SHAKE-256 uint16 per value
        (integer arithmetic, then IEEE + - * / and sqrt only, no libm), standardised
        on the pool's own sample moments."""
        if self._gauss is None:
            n = self.float_len
            g = [0] * n
            for k in range(12):
                col = struct.unpack('<%dH' % n, _seed_bytes(self.seed, 'gauss%d' % k, 2 * n))
                g = [a + b for a, b in zip(g, col)]
            mean = sum(g) / n
            std = math.sqrt(sum((x - mean) * (x - mean) for x in g) / n)
            self._gauss = [(x - mean) / std for x in g]
        return self._gauss

    def pool(self, qname, std):
        """(bytes, unit_bytes) holding whole units: elements for f32/f16, blocks otherwise."""
        key = (qname, std)
        if key in self._cache:
            return self._cache[key]
        q = QTYPES[qname]
        if qname in FLOAT_TYPES:
            vals = [x * std for x in self._standard()]
            data = struct.pack('<%d%s' % (len(vals), 'e' if qname == 'f16' else 'f'), *vals)
            res = (data, q.block_bytes)
        else:
            buf = bytearray(_seed_bytes(self.seed, qname, self.block_len * q.block_bytes))
            if qname == 'q8_0':   # the reference quantiser never writes -128
                table = bytes(129 if i == 128 else i for i in range(256))
                buf = bytearray(bytes(buf).translate(table))
            d = std / q.std_per_d
            fields = {0: d, 2: 7.5 * d} if qname == 'q4_k' else {0: d}
            for off, v in fields.items():
                b = struct.pack('<e', v)
                buf[off::q.block_bytes] = bytes([b[0]]) * self.block_len
                buf[off + 1::q.block_bytes] = bytes([b[1]]) * self.block_len
            res = (bytes(buf), q.block_bytes)
        self._cache[key] = res
        return res

    def phase(self, name, units):
        return int.from_bytes(_seed_bytes(self.seed, 'phase|' + name, 8), 'little') % units


class _Count:
    def __init__(self, f):
        self.f, self.n = f, 0

    def write(self, b):
        self.n += len(b)
        self.f.write(b)


_ZEROS = bytes(1 << 16)


def _zeros(out, n):
    while n > 0:
        k = min(n, len(_ZEROS))
        out.write(_ZEROS[:k])
        n -= k


def _write_tensor(out, t, pools):
    start = out.n
    if t.role == 'norm':
        one = struct.pack('<f', 1.0)
        left = t.elements
        while left:
            k = min(left, 1 << 14)
            out.write(one * k)
            left -= k
    else:
        zero = min(t.zero_rows * t.dims[0] // QTYPES[t.qtype].block_elems * QTYPES[t.qtype].block_bytes, t.nbytes)
        _zeros(out, zero)
        data, unit = pools.pool(t.qtype, t.std)
        mv = memoryview(data)
        plen = len(data) // unit
        pos = pools.phase(t.name, plen)
        left = (t.nbytes - zero) // unit
        while left:
            take = min(left, plen - pos)
            out.write(mv[pos * unit:(pos + take) * unit])
            pos = (pos + take) % plen
            left -= take
    assert out.n - start == t.nbytes, t.name
    _zeros(out, t.span - t.nbytes)


def write_gguf(s, fileobj, pools=None, _plan=None):
    """Stream the file for spec s into fileobj.write(); returns the byte count."""
    p = _plan or plan(s)
    pools = pools or Pools(s.seed)
    out = _Count(fileobj)
    out.write(p.header)
    _zeros(out, p.data_start - len(p.header))
    for t in p.tensors:
        _write_tensor(out, t, pools)
    assert out.n == p.file_size, (out.n, p.file_size)
    return out.n


def generate(s, path):
    """Write the file to path atomically (a .part file, renamed on success)."""
    p = plan(s)
    part = path + '.part'
    try:
        with open(part, 'wb', buffering=1 << 20) as f:
            write_gguf(s, f, _plan=p)
        os.replace(part, path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    return p


# --------------------------------------------------------------------------- cli

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arch', choices=('dense', 'moe'), required=True)
    ap.add_argument('--layers', type=int, required=True)
    ap.add_argument('--embd', type=int, required=True)
    ap.add_argument('--ff', type=int, required=True, help='dense FFN width; for moe the width of one expert')
    ap.add_argument('--heads', type=int, required=True)
    ap.add_argument('--kv-heads', type=int, default=0, help='default: same as --heads')
    ap.add_argument('--vocab', type=int, default=DEFAULT_VOCAB)
    ap.add_argument('--ctx', type=int, default=Spec.ctx)
    ap.add_argument('--experts', type=int, default=0)
    ap.add_argument('--experts-used', type=int, default=0)
    ap.add_argument('--quant', choices=tuple(QTYPES), default='f16', help='type of attention/FFN/expert matrices')
    ap.add_argument('--head-type', choices=tuple(QTYPES), default='f16', help='type of token_embd and output')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--name', default='')
    ap.add_argument('--out', help='output .gguf path')
    ap.add_argument('--plan', action='store_true', help='print the tensor table and expected size; write nothing')
    ap.add_argument('--json', action='store_true', help='with --plan: machine-readable output')
    a = ap.parse_args(argv)
    s = Spec(a.arch, a.layers, a.embd, a.ff, a.heads, a.kv_heads, a.vocab, a.ctx, a.experts, a.experts_used,
             a.quant, a.head_type, a.seed, a.name)
    try:
        p = plan(s)
    except ValueError as e:
        print('error: %s' % e, file=sys.stderr)
        return 2
    if a.plan or not a.out:
        print(json.dumps(p.to_dict(), indent=1) if a.json else format_plan(p))
        return 0
    print(format_plan(p))
    generate(s, a.out)
    got = os.path.getsize(a.out)
    if got != p.file_size:
        print('error: wrote %d bytes, planned %d' % (got, p.file_size), file=sys.stderr)
        return 1
    print('wrote %s (%d bytes)' % (a.out, got))
    return 0


if __name__ == '__main__':
    sys.exit(main())
