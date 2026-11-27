#!/usr/bin/env python3
"""Tests for tools/perf/make_synthetic_gguf.py (loadable synthetic GGUF files).

Run from the repo root:  python3 -m pytest tests/test_synthetic_gguf.py

No network, no backend, no GPU: the files are parsed by tools/gguf/parse_gguf.py
and by an independent reader below, and one numerics test runs a pure-Python
forward pass on a tiny generated file. Whether llama-server loads the files is
NOT tested here; see the later lab commands in the generator's report.
"""
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import time
import tracemalloc

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN_PATH = os.path.join(ROOT, 'tools', 'perf', 'make_synthetic_gguf.py')
PARSER_PATH = os.path.join(ROOT, 'tools', 'gguf', 'parse_gguf.py')
MANIFEST = os.path.join(ROOT, 'tools', 'perf', 'calibration_suite.json')
FIXTURE = os.path.join(ROOT, 'go', 'pkg', 'gguf', 'testdata', 'synthetic-dense-tiny.gguf.fixture')


def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _import('make_synthetic_gguf', GEN_PATH)
parser = _import('parse_gguf', PARSER_PATH)
Spec = gen.Spec

TINY = dict(arch='dense', layers=3, embd=64, ff=128, heads=2, kv_heads=1, vocab=360, ctx=512)
TINY_MOE = dict(arch='moe', layers=3, embd=64, ff=64, heads=2, kv_heads=2, vocab=360, ctx=512,
                experts=4, experts_used=2)
# Block types need every contracted width to be a multiple of the block size.
Q_WIDE = dict(layers=2, embd=256, ff=256, heads=4, kv_heads=2, vocab=360, ctx=512)

CASES = {
    'dense-f16': Spec(**TINY),
    'dense-f32': Spec(quant='f32', head_type='f32', **TINY),
    'moe-f16': Spec(**TINY_MOE),
    'dense-q8_0': Spec(arch='dense', quant='q8_0', **Q_WIDE),
    'dense-q4_0': Spec(arch='dense', quant='q4_0', **Q_WIDE),
    'moe-q4_k': Spec(arch='moe', quant='q4_k', experts=8, experts_used=2, **Q_WIDE),
    'moe-q8_0-headq8': Spec(arch='moe', quant='q8_0', head_type='q8_0', experts=4, experts_used=1, **Q_WIDE),
}


# ----------------------------------------------------------------- independent reader

class Gguf:
    """Header reader written from the GGUF spec, not from parse_gguf.py."""

    def __init__(self, path):
        self.path = path
        self.size = os.path.getsize(path)
        with open(path, 'rb') as f:
            assert f.read(4) == b'GGUF'
            self.version, self.n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
            self.kv = {}
            for _ in range(n_kv):
                key = self._str(f)
                vt = struct.unpack('<I', f.read(4))[0]
                self.kv[key] = (vt, self._value(f, vt))
            self.tensors = []
            for _ in range(self.n_tensors):
                name = self._str(f)
                nd = struct.unpack('<I', f.read(4))[0]
                dims = struct.unpack('<%dQ' % nd, f.read(8 * nd))
                ttype, off = struct.unpack('<IQ', f.read(12))
                self.tensors.append((name, dims, ttype, off))
            self.header_end = f.tell()
        align = self.kv['general.alignment'][1] if 'general.alignment' in self.kv else 32
        self.align = align
        self.data_start = (self.header_end + align - 1) // align * align

    @staticmethod
    def _str(f):
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8')

    def _value(self, f, vt):
        fixed = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
        if vt in fixed:
            fmt = '<' + fixed[vt]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if vt == 8:
            return self._str(f)
        assert vt == 9
        et, n = struct.unpack('<IQ', f.read(12))
        return [self._value(f, et) for _ in range(n)]

    def read(self, name):
        """Raw bytes of tensor `name` (without padding)."""
        for i, (n, dims, ttype, off) in enumerate(self.tensors):
            if n == name:
                nbytes = tensor_bytes(dims, ttype)
                with open(self.path, 'rb') as f:
                    f.seek(self.data_start + off)
                    return dims, ttype, f.read(nbytes)
        raise KeyError(name)


LAYOUT = {0: (4, 1), 1: (2, 1), 2: (18, 32), 8: (34, 32), 12: (144, 256)}


def tensor_bytes(dims, ttype):
    bb, be = LAYOUT[ttype]
    n = 1
    for d in dims:
        n *= d
    assert dims[0] % be == 0
    return n // be * bb


def pad32(n):
    return (n + 31) // 32 * 32


# ----------------------------------------------------------------- reference dequantisers

def f16(b):
    return struct.unpack('<e', b)[0]


def dq_q8_0(raw):  # ggml-quants.c dequantize_row_q8_0
    out = []
    for o in range(0, len(raw), 34):
        d = f16(raw[o:o + 2])
        out += [d * struct.unpack('<b', raw[o + 2 + j:o + 3 + j])[0] for j in range(32)]
    return out


def dq_q4_0(raw):  # dequantize_row_q4_0
    out = []
    for o in range(0, len(raw), 18):
        d = f16(raw[o:o + 2])
        qs = raw[o + 2:o + 18]
        out += [((q & 0xF) - 8) * d for q in qs] + [((q >> 4) - 8) * d for q in qs]
    return out


def _scale_min_k4(j, q):  # get_scale_min_k4
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def dq_q4_k(raw):  # dequantize_row_q4_K
    out = []
    for o in range(0, len(raw), 144):
        d, dmin = f16(raw[o:o + 2]), f16(raw[o + 2:o + 4])
        scales, qs = raw[o + 4:o + 16], raw[o + 16:o + 144]
        is_, q0 = 0, 0
        for _ in range(0, 256, 64):
            sc, m = _scale_min_k4(is_, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = _scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            q = qs[q0:q0 + 32]
            out += [d1 * (x & 0xF) - m1 for x in q] + [d2 * (x >> 4) - m2 for x in q]
            q0 += 32
            is_ += 2
    return out


def decode(dims, ttype, raw):
    n = 1
    for d in dims:
        n *= d
    if ttype == 0:
        return list(struct.unpack('<%df' % n, raw))
    if ttype == 1:
        return list(struct.unpack('<%de' % n, raw))
    return {2: dq_q4_0, 8: dq_q8_0, 12: dq_q4_k}[ttype](raw)


def mean_std(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ----------------------------------------------------------------- fixtures

@pytest.fixture(scope='module')
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp('synth')
    out = {}
    for name, spec in CASES.items():
        path = str(d / (name + '.gguf'))
        gen.generate(spec, path)
        out[name] = (spec, path)
    return out


# ----------------------------------------------------------------- tests

@pytest.mark.parametrize('case', sorted(CASES))
def test_plan_equals_parser(files, case):
    spec, path = files[case]
    p = gen.plan(spec)
    assert os.path.getsize(path) == p.file_size
    r = parser.parse(path)
    assert r['arch'] == 'llama'
    assert r['layers'] == spec.layers
    assert r['embd'] == spec.embd and r['ff'] == spec.ff
    assert r['heads'] == spec.heads and r['hkv'] == spec.kv
    assert r['kl'] == r['vl'] == r['n_rot'] == spec.head_dim
    assert r['ctx_train'] == spec.ctx and r['vocab_size'] == spec.vocab
    if spec.arch == 'moe':
        assert r['experts'] == spec.experts and r['exp_used'] == spec.experts_used
    else:
        assert 'experts' not in r
    # histogram: type id, tensor count, elements, bytes by on-disk span (padded)
    got = [(e['type'], e['tensors'], e['elements'], e['bytes']) for e in r['tensor_types']]
    want = [(e['type'], e['tensors'], e['elements'], e['bytes']) for e in p.type_histogram()]
    assert got == want
    assert sum(e['tensors'] for e in r['tensor_types']) == len(p.tensors)
    # every byte of the data section is accounted to exactly one tensor
    assert r['expert_bytes'] + r['non_expert_bytes'] == p.data_size
    assert r['expert_bytes'] == sum(t.span for t in p.tensors if t.role == 'expert')
    assert r['token_embd_bytes'] == next(t.span for t in p.tensors if t.name == 'token_embd.weight')
    assert r['output_bytes'] == next(t.span for t in p.tensors if t.name == 'output.weight')


@pytest.mark.parametrize('case', sorted(CASES))
def test_header_offsets_alignment_and_names(files, case):
    spec, path = files[case]
    g = Gguf(path)
    assert g.version == 3 and g.align == 32
    assert g.data_start % 32 == 0
    expected, off = [], 0
    for name, dims, ttype, o in g.tensors:
        assert len(name) < 64                       # ggml GGML_MAX_NAME
        assert o == off and o % 32 == 0             # running sum of padded sizes (gguf.cpp)
        block = LAYOUT[ttype][1]
        assert dims[0] % block == 0                 # ne[0] a multiple of the block size
        off += pad32(tensor_bytes(dims, ttype))
        expected.append(name)
    assert g.size == g.data_start + off
    assert len(set(expected)) == len(expected)


def _expected_names(spec):
    names = ['token_embd.weight', 'output_norm.weight', 'output.weight']
    for i in range(spec.layers):
        names += ['blk.%d.%s.weight' % (i, n) for n in
                  ('attn_norm', 'attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_norm')]
        if spec.arch == 'dense':
            names += ['blk.%d.%s.weight' % (i, n) for n in ('ffn_gate', 'ffn_down', 'ffn_up')]
        else:
            names += ['blk.%d.%s.weight' % (i, n) for n in
                      ('ffn_gate_inp', 'ffn_gate_exps', 'ffn_down_exps', 'ffn_up_exps')]
    return sorted(names)


@pytest.mark.parametrize('case', sorted(CASES))
def test_tensor_set_and_shapes_match_the_llama_loader(files, case):
    """The set and the ne-order shapes of src/models/llama.cpp load_arch_tensors;
    any extra or missing tensor fails the real load."""
    spec, path = files[case]
    g = Gguf(path)
    assert sorted(t[0] for t in g.tensors) == _expected_names(spec)
    d, kvd, V = spec.embd, spec.kv * spec.head_dim, spec.vocab
    shapes = {'token_embd.weight': (d, V), 'output.weight': (d, V), 'output_norm.weight': (d,)}
    for i in range(spec.layers):
        p = 'blk.%d.' % i
        shapes.update({p + 'attn_norm.weight': (d,), p + 'ffn_norm.weight': (d,),
                       p + 'attn_q.weight': (d, d), p + 'attn_k.weight': (d, kvd),
                       p + 'attn_v.weight': (d, kvd), p + 'attn_output.weight': (d, d)})
        if spec.arch == 'dense':
            shapes.update({p + 'ffn_gate.weight': (d, spec.ff), p + 'ffn_down.weight': (spec.ff, d),
                           p + 'ffn_up.weight': (d, spec.ff)})
        else:
            E = spec.experts
            shapes.update({p + 'ffn_gate_inp.weight': (d, E), p + 'ffn_gate_exps.weight': (d, spec.ff, E),
                           p + 'ffn_down_exps.weight': (spec.ff, d, E), p + 'ffn_up_exps.weight': (d, spec.ff, E)})
    assert {t[0]: tuple(t[1]) for t in g.tensors} == shapes
    for name, dims, ttype, _ in g.tensors:   # 1-D norms and the router stay F32
        if name.endswith('norm.weight') or name.endswith('ffn_gate_inp.weight'):
            assert ttype == 0


REQUIRED_KEYS = [
    # llama-model.cpp load_hparams + src/models/llama.cpp load_arch_hparams (no default)
    'general.architecture', 'llama.context_length', 'llama.embedding_length', 'llama.block_count',
    'llama.attention.layer_norm_rms_epsilon',
    # read with required=false but the llama tensors cannot be sized without them
    'llama.feed_forward_length', 'llama.attention.head_count', 'llama.attention.head_count_kv',
    'llama.rope.dimension_count', 'llama.vocab_size',
    # vocabulary: llama-vocab.cpp (SPM) and ik_llama llama-vocab.cpp
    'tokenizer.ggml.model', 'tokenizer.ggml.tokens', 'tokenizer.ggml.scores', 'tokenizer.ggml.token_type',
    'tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.unknown_token_id',
]


@pytest.mark.parametrize('case', sorted(CASES))
def test_required_metadata_and_vocab(files, case):
    spec, path = files[case]
    kv = Gguf(path).kv
    for k in REQUIRED_KEYS:
        assert k in kv, k
    assert kv['general.architecture'][1] == 'llama'
    assert kv['tokenizer.ggml.model'][1] == 'llama'
    toks = kv['tokenizer.ggml.tokens'][1]
    assert len(toks) == spec.vocab == len(kv['tokenizer.ggml.scores'][1]) == len(kv['tokenizer.ggml.token_type'][1])
    assert len(set(toks)) == len(toks) and all(toks)             # unique, non-empty (loader assert)
    assert toks[:3] == ['<unk>', '<s>', '</s>']
    assert all(('<0x%02X>' % b) in toks for b in range(256))   # SPM byte fallback
    assert kv['llama.vocab_size'][1] == spec.vocab
    if spec.arch == 'moe':
        assert kv['llama.expert_count'][1] == spec.experts
        assert kv['llama.expert_used_count'][1] == spec.experts_used
    else:
        assert 'llama.expert_count' not in kv


def test_determinism_and_seed_sensitivity(tmp_path):
    spec = CASES['moe-q4_k']
    a, b, c = (str(tmp_path / n) for n in 'abc')
    gen.generate(spec, a)
    gen.generate(spec, b)
    gen.generate(Spec(**{**spec.__dict__, 'seed': 2}), c)
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    assert sha(a) == sha(b)
    assert sha(a) != sha(c)
    assert os.path.getsize(a) == os.path.getsize(c)
    assert not os.path.exists(a + '.part')


def test_committed_go_fixture_is_the_current_output(tmp_path):
    out = str(tmp_path / 'f.gguf')
    gen.generate(Spec(arch='dense', layers=2, embd=64, ff=128, heads=2, vocab=360, ctx=512, seed=1,
                      name='ggrun-synth-fixture'), out)
    assert open(out, 'rb').read() == open(FIXTURE, 'rb').read()


def test_unordered_phase_and_pools_are_independent_of_call_order():
    a, b = gen.Pools(5, 4099, 257), gen.Pools(5, 4099, 257)
    x1, y1 = a.pool('f16', 0.5), a.pool('q8_0', 0.5)
    y2, x2 = b.pool('q8_0', 0.5), b.pool('f16', 0.5)
    assert x1 == x2 and y1 == y2


def test_pool_lengths_are_prime():
    def prime(n):
        return n > 1 and all(n % i for i in range(2, int(n ** 0.5) + 1))
    assert prime(gen.FLOAT_POOL) and prime(gen.BLOCK_POOL)


def test_block_layouts_match_the_parsers():
    for q in gen.QTYPES.values():
        assert parser._LAYOUT_TYPES[q.id] == (q.block_bytes, q.block_elems), q.name


@pytest.mark.parametrize('qname', ['q4_0', 'q8_0', 'q4_k'])
def test_quantised_values_are_finite_with_the_target_std(qname):
    q = gen.QTYPES[qname]
    std = 0.05
    pools = gen.Pools(3, 1031, 1031)
    data, unit = pools.pool(qname, std)
    assert unit == q.block_bytes and len(data) == 1031 * unit
    vals = decode((1031 * q.block_elems,), q.id, data)
    assert all(math.isfinite(v) for v in vals)
    m, s = mean_std(vals)
    assert abs(s / std - 1) < 0.06, (qname, s)
    assert abs(m) < 0.15 * std, (qname, m)       # Q4_0 keeps the reference grid's -0.5 offset
    if qname == 'q8_0':
        assert b'\x80' not in b''.join(data[o + 2:o + 34] for o in range(0, len(data), 34))


def test_float_pool_is_exactly_standardised():
    pools = gen.Pools(9, 2003, 11)
    for name, fmt in (('f32', '<%df'), ('f16', '<%de')):
        data, _ = pools.pool(name, 0.25)
        m, s = mean_std(struct.unpack(fmt % 2003, data))
        assert abs(m) < 2e-3 and abs(s - 0.25) < 2e-3


def test_special_token_rows_of_the_output_head_are_zero(files):
    for case in ('dense-f16', 'dense-q8_0', 'moe-q4_k'):
        spec, path = files[case]
        dims, ttype, raw = Gguf(path).read('output.weight')
        row = tensor_bytes((dims[0],), ttype)
        assert raw[:3 * row] == bytes(3 * row)
        assert any(raw[3 * row:4 * row])


# ---- forward pass: finite activations, EOS never the argmax ---------------------

def _matvec(rows, x):
    return [sum(a * b for a, b in zip(r, x)) for r in rows]


def _rows(vals, ne0):
    return [vals[i:i + ne0] for i in range(0, len(vals), ne0)]


def _rms(x, w):
    k = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + gen.EPS)
    return [v * k * g for v, g in zip(x, w)]


def _load(g, name):
    dims, ttype, raw = g.read(name)
    return dims, decode(dims, ttype, raw)


def forward_one_token(g, spec, tok):
    """One position, attention reduced to its value path (softmax over one
    position is 1). That is the worst case for growth: with more positions the
    attention average only shrinks the update."""
    d, hd, H, KV = spec.embd, spec.head_dim, spec.heads, spec.kv
    dims, emb = _load(g, 'token_embd.weight')
    x = emb[tok * d:(tok + 1) * d]
    peak = max(abs(v) for v in x)
    for i in range(spec.layers):
        p = 'blk.%d.' % i
        h = _rms(x, _load(g, p + 'attn_norm.weight')[1])
        v = _matvec(_rows(_load(g, p + 'attn_v.weight')[1], d), h)
        attn = [v[(j // (H // KV)) * hd + k] for j in range(H) for k in range(hd)]
        x = [a + b for a, b in zip(x, _matvec(_rows(_load(g, p + 'attn_output.weight')[1], d), attn))]
        h = _rms(x, _load(g, p + 'ffn_norm.weight')[1])
        ff = spec.ff

        def ffn(gate, up, down):
            a = [(gv / (1 + math.exp(-gv))) * uv for gv, uv in zip(_matvec(gate, h), _matvec(up, h))]
            return _matvec(down, a)
        if spec.arch == 'dense':
            y = ffn(_rows(_load(g, p + 'ffn_gate.weight')[1], d), _rows(_load(g, p + 'ffn_up.weight')[1], d),
                    _rows(_load(g, p + 'ffn_down.weight')[1], ff))
        else:
            logits = _matvec(_rows(_load(g, p + 'ffn_gate_inp.weight')[1], d), h)
            mx = max(logits)
            probs = [math.exp(v - mx) for v in logits]
            top = sorted(range(spec.experts), key=lambda e: -probs[e])[:spec.experts_used]
            z = sum(probs[e] for e in top)
            gate, up, down = (_load(g, p + n)[1] for n in ('ffn_gate_exps.weight', 'ffn_up_exps.weight',
                                                           'ffn_down_exps.weight'))
            y = [0.0] * d
            for e in top:
                ye = ffn(_rows(gate[e * ff * d:(e + 1) * ff * d], d), _rows(up[e * ff * d:(e + 1) * ff * d], d),
                         _rows(down[e * d * ff:(e + 1) * d * ff], ff))
                y = [a + probs[e] / z * b for a, b in zip(y, ye)]
        x = [a + b for a, b in zip(x, y)]
        peak = max(peak, max(abs(v) for v in x))
    h = _rms(x, _load(g, 'output_norm.weight')[1])
    out = _load(g, 'output.weight')[1]
    return _matvec(_rows(out, d), h), peak


@pytest.mark.parametrize('case', ['dense-f16', 'moe-f16', 'moe-q8_0-headq8', 'dense-q4_0', 'moe-q4_k'])
def test_forward_pass_stays_finite_and_eos_is_never_argmax(files, case):
    spec, path = files[case]
    g = Gguf(path)
    for tok in (1, 17, 258, 300, 359):
        logits, peak = forward_one_token(g, spec, tok)
        assert all(math.isfinite(v) for v in logits)
        assert peak < 50.0, peak                              # residual stream: no blow-up
        assert logits[:3] == [0.0, 0.0, 0.0]                  # <unk>, <s>, </s>
        m, s = mean_std(logits[3:])
        assert 3.0 < s < 12.0, s                              # about logit_std = 6
        assert max(range(len(logits)), key=lambda i: logits[i]) >= 3


def test_residual_growth_is_sublinear_in_depth(tmp_path):
    """Deep model, F16: the residual stream stays far from F16/F32 limits."""
    spec = Spec(arch='dense', layers=24, embd=64, ff=128, heads=2, kv_heads=2, vocab=360, ctx=512)
    path = str(tmp_path / 'deep.gguf')
    gen.generate(spec, path)
    logits, peak = forward_one_token(Gguf(path), spec, 100)
    assert all(math.isfinite(v) for v in logits)
    assert peak < 60.0


# ---- bounded memory, big plans ---------------------------------------------------

class _Sink:
    def __init__(self):
        self.n = 0

    def write(self, b):
        self.n += len(b)


def test_streaming_writer_memory_is_bounded():
    spec = Spec(arch='moe', layers=4, embd=1024, ff=1024, heads=8, kv_heads=4, vocab=512,
                experts=16, experts_used=2, quant='f16')
    p = gen.plan(spec)
    assert p.file_size > 300 * 2 ** 20
    sink = _Sink()
    tracemalloc.start()
    try:
        n = gen.write_gguf(spec, sink, pools=gen.Pools(1))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert n == sink.n == p.file_size
    assert peak < 48 * 2 ** 20, peak           # pools and chunks, not the file


def test_plan_of_a_70b_class_model_is_instant_and_exact():
    spec = Spec(arch='dense', layers=80, embd=8192, ff=28672, heads=64, kv_heads=8, vocab=32000, quant='f16')
    t0 = time.time()
    p = gen.plan(spec)
    assert time.time() - t0 < 2.0
    d, kv = 8192, 8 * 128
    per_layer = (d * d + 2 * d * kv + d * d + 3 * d * 28672) * 2 + 2 * d * 4
    assert p.data_size == per_layer * 80 + 2 * 32000 * d * 2 + d * 4
    assert p.file_size == p.data_start + p.data_size
    assert p.file_size > 100 * 2 ** 30


def test_dense_moe_pair_reads_equal_attention_and_ffn_bytes():
    dense = gen.plan(Spec(arch='dense', layers=4, embd=256, ff=1024, heads=4, kv_heads=2, quant='q4_k'))
    moe = gen.plan(Spec(arch='moe', layers=4, embd=256, ff=256, heads=4, kv_heads=2, experts=16, experts_used=4,
                        quant='q4_k'))
    router = sum(t.nbytes for t in moe.tensors if t.role == 'router')
    assert moe.active_weight_bytes - router == dense.active_weight_bytes


# ---- validation and CLI ------------------------------------------------------------

@pytest.mark.parametrize('bad', [
    dict(embd=65), dict(heads=3), dict(kv_heads=3), dict(vocab=100), dict(quant='q4_k'),
    dict(arch='moe'), dict(arch='moe', experts=4, experts_used=5), dict(experts=2, experts_used=1),
    dict(quant='bf16'),
])
def test_invalid_specs_are_refused(bad):
    base = dict(arch='dense', layers=2, embd=64, ff=128, heads=2, vocab=360)
    base.update(bad)
    with pytest.raises(ValueError):
        gen.plan(Spec(**base))


def _cli(*args):
    return subprocess.run([sys.executable, '-I', GEN_PATH] + list(args), capture_output=True, text=True)


def test_cli_plan_prints_the_size_and_writes_nothing(tmp_path):
    args = ['--arch', 'moe', '--layers', '2', '--embd', '64', '--ff', '64', '--heads', '2', '--experts', '4',
            '--experts-used', '2', '--vocab', '360']
    r = _cli(*args, '--plan')
    assert r.returncode == 0, r.stderr
    p = gen.plan(Spec(arch='moe', layers=2, embd=64, ff=64, heads=2, experts=4, experts_used=2, vocab=360))
    assert 'FILE SIZE=%d B' % p.file_size in r.stdout
    assert 'blk.1.ffn_up_exps.weight' in r.stdout
    j = json.loads(_cli(*args, '--plan', '--json').stdout)
    assert j['file_size'] == p.file_size and j['tensor_count'] == len(p.tensors)
    out = str(tmp_path / 'm.gguf')
    r = _cli(*args, '--out', out)
    assert r.returncode == 0, r.stderr
    assert os.path.getsize(out) == p.file_size


def test_cli_refuses_a_bad_shape():
    r = _cli('--arch', 'dense', '--layers', '2', '--embd', '300', '--ff', '256', '--heads', '2',
             '--quant', 'q4_k', '--plan')
    assert r.returncode == 2 and 'multiple of the q4_k block size' in r.stderr


# ---- the calibration suite manifest -------------------------------------------------

def test_manifest_sizes_equal_plan_and_the_suite_is_small():
    m = json.load(open(MANIFEST))
    ids = [f['id'] for f in m['files']]
    assert len(set(ids)) == len(ids) >= 8
    total = quick = 0
    for f in m['files']:
        assert f['tier'] in ('quick', 'full') and f['isolates'] and f['compare'] and f['from']
        assert f['spec']['name'] == 'ggrun-synth-' + f['id']
        p = gen.plan(Spec(**f['spec']))
        assert p.file_size == f['expected_file_bytes'], f['id']
        assert p.active_weight_bytes == f['active_weight_bytes'], f['id']
        assert p.kv_bytes_per_token_f16 == f['kv_bytes_per_token_f16'], f['id']
        assert p.file_size <= 2 * 2 ** 30, f['id']            # each file loads in seconds
        total += p.file_size
        quick += p.file_size if f['tier'] == 'quick' else 0
    assert total == m['total_bytes'] and quick == m['quick_tier_bytes']
    assert total <= 8 * 2 ** 30 and quick <= 2 * 2 ** 30


def test_manifest_has_the_requested_shapes():
    m = {f['id']: f['spec'] for f in json.load(open(MANIFEST))['files']}
    for need in ('fixed-floor', 'layers-48', 'bytes-large-1', 'bytes-mid', 'pair-dense', 'pair-moe',
                 'depth-kv', 'type-q8_0', 'moe-realshape'):
        assert need in m, need
    a, c, big = m['fixed-floor'], m['layers-48'], m['bytes-large-1']
    assert (a['embd'], a['ff']) == (c['embd'], c['ff']) and c['layers'] > a['layers']       # layers axis
    assert big['layers'] == a['layers'] and big['embd'] > m['bytes-mid']['embd'] > a['embd']  # width axis
    d, e = m['pair-dense'], m['pair-moe']
    assert d['ff'] == e['ff'] * e['experts_used']                                             # equal active FFN width
    assert all(d[k] == e[k] for k in ('layers', 'embd', 'heads', 'kv_heads'))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
