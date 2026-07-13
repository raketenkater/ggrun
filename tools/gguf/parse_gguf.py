#!/usr/bin/env python3
"""GGUF metadata parser for ggrun.

Extracts architecture, layer counts, expert layout, KV geometry, and tensor
byte totals. All fields needed by ggrun for placement and RAM estimation.

Usage:
    parse_gguf.py [--format json|shell] MODEL_PATH

In shell mode, emits `VAR=value` lines safe for `eval "$(parse_gguf.py --format shell ...)"`.
Variable names match the ggrun bash script's expectations.

Importable API:
    from parse_gguf import parse
    metadata = parse('/path/to/model.gguf')
"""
import argparse
import hashlib
import json
import os
import re
import struct
import sys
from typing import Any, Dict

# (bytes_per_block, elements_per_block) per ggml type id — from ggml.h struct sizes.
# IDs 0–31 are upstream llama.cpp; 137+ are ik_llama.cpp custom quants
# (IQ2_K through IQ6_K + KS/KSS/KT/KL/KL variants and 337+ _R4 row-quantized
# rearrangements that share bpw with their base type). Without these, the
# fallback below treats every unknown type as F16 (2 B/elem) and
# over-estimates expert tensor bytes 3-5x — the cause of the "313% expert"
# RAM-fit bug for ik_llama-quantized MoE models like Kimi-K2.
GGUF_TYPE_SIZE = {
    0: (4, 1), 1: (2, 1),
    2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32),
    8: (34, 32), 9: (36, 32), 10: (20, 32), 11: (36, 64),
    12: (144, 256), 13: (176, 256), 14: (210, 256), 15: (292, 256),
    16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256),
    20: (18, 32), 21: (110, 256), 22: (82, 256), 23: (136, 256),
    24: (56, 256), 25: (2, 1), 26: (18, 32), 27: (18, 32),
    28: (18, 32), 29: (40, 256), 30: (54, 256), 31: (1, 1),
    39: (17, 32),   # MXFP4 — 1 scale byte + 16 nibble-packed bytes per 32 elems
    # ik_llama.cpp custom quants
    137: (76, 256),    # IQ2_K   — 2.375 bpw
    138: (110, 256),   # IQ3_K   — 3.44 bpw
    139: (144, 256),   # IQ4_K   — 4.5 bpw
    140: (176, 256),   # IQ5_K   — 5.5 bpw
    141: (212, 256),   # IQ6_K   — 6.625 bpw
    144: (136, 256),   # IQ4_KS
    145: (70, 256),    # IQ2_KS
    146: (128, 256),   # IQ4_KSS
    152: (168, 256),   # IQ5_KS
    153: (68, 256),    # IQ2_KT
    154: (100, 256),   # IQ3_KT
    155: (128, 256),   # IQ4_KT
    156: (102, 256),   # IQ3_KS
    157: (86, 256),    # IQ2_KL
    # _R4 row-quantized: 4 rows packed; bytes-per-element matches the base
    337: (76, 256),    # IQ2_K_R4
    338: (110, 256),   # IQ3_K_R4
    339: (144, 256),   # IQ4_K_R4
    340: (176, 256),   # IQ5_K_R4
    344: (136, 256),   # IQ4_KS_R4
    352: (168, 256),   # IQ5_KS_R4
}

# GGUF value-type fixed sizes. 8=string, 9=array are variable-length.
_KV_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_kv(f, r, kv_count):
    for _ in range(kv_count):
        kl = struct.unpack('<Q', f.read(8))[0]
        key = f.read(kl).decode('utf-8', errors='replace')
        vt = struct.unpack('<I', f.read(4))[0]
        if vt == 4:  # uint32
            val = struct.unpack('<I', f.read(4))[0]
            if key.endswith('.block_count'): r['layers'] = val
            if 'expert_count' in key and 'used' not in key: r['experts'] = val
            if key.endswith('.expert_used_count'): r['exp_used'] = val
            if 'head_count_kv' in key: r['hkv'] = val
            if key.endswith('.attention.key_length'): r['kl'] = val
            if key.endswith('.attention.value_length'): r['vl'] = val
            if key.endswith('.attention.key_length_mla'): r['kl_mla'] = val
            if key.endswith('.attention.value_length_mla'): r['vl_mla'] = val
            if 'ssm.state_size' in key: r['ssm'] = 1
            if key.endswith('.embedding_length'): r['embd'] = val
            if key.endswith('.feed_forward_length'): r['ff'] = val
            if key.endswith('.expert_feed_forward_length'): r['exp_ff'] = val
            if key.endswith('.expert_shared_feed_forward_length'): r['exp_shared_ff'] = val
            if key.endswith('.attention.kv_lora_rank'): r['kv_lora'] = val
            if key.endswith('.attention.q_lora_rank'): r['q_lora'] = val
            if key.endswith('.rope.dimension_count'): r['n_rot'] = val
            if key.endswith('.leading_dense_block_count'): r['leading_dense'] = val
            if key.endswith('.attention.sliding_window'): r['swa'] = val
            if key.endswith('.full_attention_interval') or key.endswith('.attention.full_attention_interval'):
                r['full_interval'] = val
            if key.endswith('.context_length'): r['ctx_train'] = val
            if key.endswith('.nextn_predict_layers'): r['nextn_predict_layers'] = val
            if key == 'general.alignment': r['_align'] = val
        elif vt == 8:  # string
            sl = struct.unpack('<Q', f.read(8))[0]
            val = f.read(sl).decode('utf-8', errors='replace')
            if key == 'general.architecture': r['arch'] = val
            elif key == 'general.name': r['name'] = val
            elif key == 'general.basename': r['basename'] = val
            elif key == 'general.quantized_by': r['quantized_by'] = val
            elif key == 'tokenizer.ggml.model': r['tokenizer_model'] = val
            elif key == 'tokenizer.ggml.pre': r['tokenizer_pre'] = val
        elif vt == 9:  # array
            at = struct.unpack('<I', f.read(4))[0]
            al = struct.unpack('<Q', f.read(8))[0]
            if key == 'tokenizer.ggml.tokens':
                r['vocab_size'] = al
            if at in _KV_FIXED:
                f.read(al * _KV_FIXED[at])
            elif at == 8:
                token_hash = hashlib.sha256() if key == 'tokenizer.ggml.tokens' else None
                for _ in range(al):
                    length_raw = f.read(8)
                    length = struct.unpack('<Q', length_raw)[0]
                    value = f.read(length)
                    if token_hash is not None:
                        # Length framing prevents ambiguous concatenations such
                        # as ["ab", "c"] and ["a", "bc"] from colliding.
                        token_hash.update(length_raw)
                        token_hash.update(value)
                if token_hash is not None:
                    r['tokenizer_hash'] = token_hash.hexdigest()
            else:
                return  # nested or unknown — we've already captured what we need
        elif vt in _KV_FIXED:
            f.read(_KV_FIXED[vt])
        else:
            return


def _read_tensors(f, r, tensor_count):
    """Read the tensor table of one shard. Returns a list of
    (name, data_offset, type_math_bytes) headers, or a partial list if the
    table is truncated/corrupt. Byte accounting happens in _account_tensors
    once real on-disk spans are known."""
    tensors = []
    for _ in range(tensor_count):
        try:
            tl = struct.unpack('<Q', f.read(8))[0]
            tname = f.read(tl).decode('utf-8', errors='replace')
            if 'ffn_up_gate' in tname or 'ffn_gate_up' in tname:
                r['fused'] = 1
            if '_shexp' in tname or '_chexp' in tname:
                r['has_shexp'] = 1
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            ttype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            n_elements = 1
            for d in dims:
                n_elements *= d
            if ttype in GGUF_TYPE_SIZE:
                bpb, epb = GGUF_TYPE_SIZE[ttype]
                n_blocks = (n_elements + epb - 1) // epb
                tbytes = n_blocks * bpb
            else:
                # Unknown ttype — could be a brand-new quant or a backend-
                # specific format. Default 0.5 B/elem (~4 bpw) as the typical
                # quant midpoint; the span sizing below replaces this estimate
                # with the real on-disk bytes whenever offsets are usable.
                # Track unknown types so callers can warn.
                tbytes = n_elements // 2
                r.setdefault('unknown_ttypes', set()).add(ttype)
            tensors.append((tname, offset, tbytes))
        except Exception:
            break
    return tensors


def _account_tensors(r, tensors, header_end, file_size, align):
    """Accumulate expert/non-expert byte totals for one shard.

    Primary sizing is the tensor's real on-disk span (delta between sorted
    data offsets; the last tensor runs to end-of-file). This is exact for
    every quant type — including ones the GGUF_TYPE_SIZE table has never
    heard of — and includes the inter-tensor alignment padding that actually
    occupies memory when loaded. Type-math is the fallback when a shard's
    offsets are unusable (out of order, overlapping, or past end of file).
    Under-counting here is what once let placement plan one expert layer too
    many and CUDA-OOM after a 15-minute model load."""
    if not tensors:
        return
    if align <= 0:
        align = 32
    data_start = (header_end + align - 1) // align * align
    data_size = file_size - data_start

    by_offset = sorted(tensors, key=lambda x: x[1])
    span_ok = data_size > 0 and by_offset[0][1] == 0
    spans = {}
    if span_ok:
        for i, (tname, off, _) in enumerate(by_offset):
            end = by_offset[i + 1][1] if i + 1 < len(by_offset) else data_size
            if end <= off:
                span_ok = False
                break
            spans[tname] = end - off

    for tname, _, tbytes in tensors:
        nbytes = spans[tname] if span_ok else tbytes
        is_expert = '_exps.' in tname or '_shexp.' in tname or 'experts.' in tname
        if is_expert:
            r['expert_bytes'] = r.get('expert_bytes', 0) + nbytes
            if '_shexp.' in tname:
                # Shared experts ride with their layer's device: the `exps=CPU`
                # -ot catch-all does not match "shexp", so CPU-offloaded layers
                # still keep their shared expert on the owning GPU. Placement
                # needs this split to budget VRAM and RAM correctly.
                r['shexp_bytes'] = r.get('shexp_bytes', 0) + nbytes
        else:
            r['non_expert_bytes'] = r.get('non_expert_bytes', 0) + nbytes
            if tname == 'token_embd.weight':
                # Input embeddings stay in host memory (llama.cpp keeps the
                # input layer on CPU), so placement must not charge them
                # against GPU VRAM budgets.
                r['token_embd_bytes'] = r.get('token_embd_bytes', 0) + nbytes
            elif tname == 'output.weight':
                # The output head lands on the device that owns the last
                # layer slot (llama.cpp splits n_layer+1 slots across the
                # tensor-split), not pro-rata across all GPUs.
                r['output_bytes'] = r.get('output_bytes', 0) + nbytes


def parse(path: str) -> Dict[str, Any]:
    """Parse a GGUF file and return extracted metadata as a dict.

    Missing keys mean the GGUF didn't expose that metadata. Numeric keys are
    int, strings are str. Consumers should `.get(key, default)` rather than
    index directly.
    """
    r: Dict[str, Any] = {'fused': 0, 'expert_bytes': 0, 'non_expert_bytes': 0}

    def read_shard(sp: str, meta: Dict[str, Any]) -> None:
        with open(sp, 'rb') as f:
            if f.read(4) != b'GGUF':
                return
            f.read(4)  # version
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            kv_count = struct.unpack('<Q', f.read(8))[0]
            _read_kv(f, meta, kv_count)
            tensors = _read_tensors(f, r, tensor_count)
            if len(tensors) < tensor_count:
                # Truncated tensor table: spans would swallow unread tensors'
                # bytes. Account what we have via type-math only (file_size 0
                # makes _account_tensors reject spans).
                _account_tensors(r, tensors, 0, 0, 32)
                return
            _account_tensors(r, tensors, f.tell(), os.path.getsize(sp),
                             meta.get('_align', 32))

    try:
        read_shard(path, r)
    except Exception:
        return r

    # Split GGUF: scan sibling shards for tensor totals. KV metadata is
    # duplicated across shards so we skip it on the non-first shards.
    m = re.search(r'-(\d+)-of-(\d+)\.gguf$', path)
    if m:
        total = int(m.group(2))
        base = path[:m.start()]
        for sn in range(2, total + 1):
            sp = f'{base}-{sn:05d}-of-{total:05d}.gguf'
            if not os.path.exists(sp):
                continue
            try:
                read_shard(sp, {})
            except Exception:
                continue
    r.pop('_align', None)
    return r


# (metadata key, bash variable name, default-for-missing)
SHELL_KEY_MAP = [
    ('layers',            'LAYER_COUNT',         0),
    ('experts',           'EXPERT_COUNT',        0),
    ('hkv',               'HEAD_COUNT_KV',       0),
    ('kl',                'KEY_LENGTH',          0),
    ('vl',                'VALUE_LENGTH',        0),
    ('kl_mla',            'KEY_LENGTH_MLA',      0),
    ('vl_mla',            'VALUE_LENGTH_MLA',    0),
    ('ssm',               'HAS_SSM',             0),
    ('fused',             'HAS_FUSED',           0),
    ('expert_bytes',      'EXPERT_BYTES',        0),
    ('non_expert_bytes',  'NON_EXPERT_BYTES',    0),
    ('token_embd_bytes',  'TOKEN_EMBD_BYTES',    0),
    ('output_bytes',      'OUTPUT_BYTES',        0),
    ('shexp_bytes',       'SHEXP_BYTES',         0),
    ('arch',              'MODEL_ARCH',          'unknown'),
    ('embd',              'EMBEDDING_LENGTH',    0),
    ('ff',                'FEED_FORWARD_LENGTH', 0),
    ('exp_used',          'EXPERT_USED_COUNT',   0),
    ('exp_ff',            'EXPERT_FF',           0),
    ('exp_shared_ff',     'EXPERT_SHARED_FF',    0),
    ('kv_lora',           'KV_LORA_RANK',        0),
    ('q_lora',            'Q_LORA_RANK',         0),
    ('n_rot',             'ROPE_DIM',            0),
    ('leading_dense',     'LEADING_DENSE',       0),
    ('swa',               'SLIDING_WINDOW',      0),
    ('full_interval',     'FULL_ATTN_INTERVAL',  0),
    ('has_shexp',         'HAS_SHEXP',           0),
    ('ctx_train',         'CTX_TRAIN',           0),
    ('nextn_predict_layers', 'NEXTN_PREDICT_LAYERS', 0),
    ('name',              'GGUF_MODEL_NAME',     ''),
    ('basename',          'GGUF_BASENAME',       ''),
    ('quantized_by',      'GGUF_QUANTIZED_BY',   ''),
    ('tokenizer_model',   'GGUF_TOKENIZER_MODEL', ''),
    ('tokenizer_pre',     'GGUF_TOKENIZER_PRE',  ''),
    ('tokenizer_hash',    'GGUF_TOKENIZER_HASH', ''),
    ('vocab_size',        'GGUF_VOCAB_SIZE',     0),
]


def _shell_quote(v: Any) -> str:
    if isinstance(v, int):
        return str(v)
    s = str(v)
    return "'" + s.replace("'", "'\\''") + "'"


def _emit_shell(r: Dict[str, Any]) -> None:
    for key, var, default in SHELL_KEY_MAP:
        val = r.get(key, default)
        print(f'{var}={_shell_quote(val)}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Parse GGUF metadata for ggrun.')
    ap.add_argument('path', help='Path to .gguf file (first shard for split models)')
    ap.add_argument('--format', choices=['json', 'shell'], default='json',
                    help='Output format: json (default) or shell VAR=value lines')
    args = ap.parse_args()
    r = parse(args.path)
    if 'unknown_ttypes' in r:
        r['unknown_ttypes'] = sorted(r['unknown_ttypes'])
    if args.format == 'json':
        json.dump(r, sys.stdout)
        sys.stdout.write('\n')
    else:
        _emit_shell(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
