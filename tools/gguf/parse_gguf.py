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

# Integer (and bool) GGUF element types, as struct formats. llama.cpp's loader
# accepts bool/uint32/int32 per-layer arrays; the others are read so a valid
# file with a wider encoding is not silently dropped.
_INT_FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 7: 'B', 10: 'Q', 11: 'q'}

# Per-layer attention arrays. A mixed-head model (MiMo-V2) states its KV head
# count and sliding-window layout per block; skipping them left ggrun with no
# KV width at all. Values are validated here only for type and sign: whether
# the length matches the block count is placement's call, because it also
# knows which blocks the backend actually caches.
_LAYER_ARRAYS = {
    '.attention.head_count_kv': 'hkv_arr',
    '.attention.sliding_window_pattern': 'swa_pattern',
}

# Recurrent state that llama.cpp keeps beside the KV cache
# (llm_arch_is_recurrent / llm_arch_is_hybrid). The backend cannot truncate
# it, so a request that branches inside a cached prompt resumes only from a
# context checkpoint. 'ssm' remains the SSM block-layout flag; 'recurrent' is
# the cache semantics. GLM-5.3-Flash states only ssm.conv_kernel and kda.*,
# and Inkling only shortconv_kernel; neither has ssm.state_size.
_RECURRENT_KEY_MARKERS = ('.ssm.', '.kda.', '.shortconv', '.wkv.')
_RECURRENT_ARCHS = frozenset({
    'mamba', 'mamba2', 'rwkv6', 'rwkv6qwen2', 'rwkv7', 'arwkv7',
    'jamba', 'falcon-h1', 'plamo2', 'granitehybrid', 'lfm2', 'lfm2moe',
    'nemotron_h', 'nemotron_h_moe', 'qwen3next', 'kimi-linear', 'bailingmoe3',
    'kimi-k3', 'qwen35', 'qwen35moe', 'qwen4exp', 'deepseek4', 'minimax-01',
    'glm5next', 'inkling',
})


def _read_int_array(f, at, al):
    fmt = _INT_FORMATS[at]
    size = struct.calcsize('<' + fmt)
    raw = f.read(al * size)
    if len(raw) != al * size:
        raise EOFError('truncated array')
    return list(struct.unpack('<%d%s' % (al, fmt), raw))


def _read_kv(f, r, kv_count):
    for _ in range(kv_count):
        kl = struct.unpack('<Q', f.read(8))[0]
        key = f.read(kl).decode('utf-8', errors='replace')
        vt = struct.unpack('<I', f.read(4))[0]
        if any(marker in key for marker in _RECURRENT_KEY_MARKERS):
            r['recurrent'] = 1
        if vt == 4:  # uint32
            val = struct.unpack('<I', f.read(4))[0]
            if key.endswith('.block_count'): r['layers'] = val
            # Looped transformers (Nanbeige4.2) run the physical blocks num_loops
            # times and keep a KV cache per pass. The older loop_count spelling
            # already reports the logical block count, so only num_loops scales KV.
            if key.endswith('.num_loops'): r['kv_loops'] = val
            if 'expert_count' in key and 'used' not in key: r['experts'] = val
            if key.endswith('.expert_used_count'): r['exp_used'] = val
            if 'head_count_kv' in key: r['hkv'] = val
            # Attention head count is a standard GGUF key, not something only
            # derivable from key_length. A model that states head_count but
            # omits key_length (stories260K, and older converts generally) left
            # ggrun with no head width at all, so the KV block-size guard could
            # not see that q8_0 was impossible. Substring 'head_count_kv' above
            # cannot match this key, so the two do not collide.
            if key.endswith('.attention.head_count'): r['heads'] = val
            if key.endswith('.attention.key_length'): r['kl'] = val
            if key.endswith('.attention.value_length'): r['vl'] = val
            # Windowed layers may use their own head width; the backend
            # defaults these to the full-attention widths when absent.
            if key.endswith('.attention.key_length_swa'): r['kl_swa'] = val
            if key.endswith('.attention.value_length_swa'): r['vl_swa'] = val
            if key.endswith('.attention.key_length_mla'): r['kl_mla'] = val
            if key.endswith('.attention.value_length_mla'): r['vl_mla'] = val
            if 'ssm.state_size' in key: r['ssm'] = 1
            # Recurrent-state geometry: the backend keeps this per slot on
            # each recurrent block's device, beside the attention KV.
            if key.endswith('.ssm.conv_kernel'): r['ssm_d_conv'] = val
            if key.endswith('.ssm.state_size'): r['ssm_d_state'] = val
            if key.endswith('.ssm.group_count'): r['ssm_n_group'] = val
            if key.endswith('.ssm.inner_size'): r['ssm_d_inner'] = val
            if key.endswith('.ssm.time_step_rank'): r['ssm_dt_rank'] = val
            if key.endswith('.embedding_length'): r['embd'] = val
            if key.endswith('.feed_forward_length'): r['ff'] = val
            if key.endswith('.expert_feed_forward_length'): r['exp_ff'] = val
            if key.endswith('.expert_shared_feed_forward_length'): r['exp_shared_ff'] = val
            if key.endswith('.expert_shared_count'): r['expert_shared_count'] = val
            if key.endswith('.attention.kv_lora_rank'): r['kv_lora'] = val
            if key.endswith('.attention.q_lora_rank'): r['q_lora'] = val
            if key.endswith('.rope.dimension_count'): r['n_rot'] = val
            if key.endswith('.leading_dense_block_count'):
                r['leading_dense'] = val
                r['_leading_dense_metadata'] = 1
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
            layer_key = next((v for k, v in _LAYER_ARRAYS.items() if key.endswith(k)), None)
            if layer_key and at in _INT_FORMATS and 0 < al <= 4096:
                values = _read_int_array(f, at, al)
                if layer_key == 'swa_pattern':
                    # The loader reads any non-zero entry as "windowed".
                    r[layer_key] = [1 if v != 0 else 0 for v in values]
                elif all(v >= 0 for v in values):
                    r[layer_key] = values
                    # A uniform array is exactly the scalar form; publish it so
                    # scalar consumers keep working. Mixed counts are never
                    # averaged into a scalar.
                    if values and all(v == values[0] for v in values) and values[0] > 0:
                        r['hkv'] = values[0]
            elif at in _KV_FIXED:
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
            # Some HY3 GGUFs omit metadata that the current reviewed fork
            # requires. The tensor layout is authoritative: routed layers have
            # ffn_gate_inp, and the shared projection's FF axis is an exact
            # multiple of one routed expert's FF size.
            moe_layer = re.match(r'^blk\.(\d+)\.ffn_gate_inp\.weight$', tname)
            if moe_layer:
                layer = int(moe_layer.group(1))
                r['_first_moe_layer'] = min(r.get('_first_moe_layer', layer), layer)
            if (tname.endswith('.ffn_gate_shexp.weight') and
                    'expert_shared_count' not in r and r.get('exp_ff', 0) > 0 and
                    len(dims) >= 2 and dims[1] % r['exp_ff'] == 0):
                count = dims[1] // r['exp_ff']
                if count > 0:
                    r['_inferred_expert_shared_count'] = count
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

    def add_layer_bytes(key, layer, nbytes):
        if layer is None:
            return
        values = r.setdefault(key, {})
        values[layer] = values.get(layer, 0) + nbytes

    for tname, _, tbytes in tensors:
        nbytes = spans[tname] if span_ok else tbytes
        layer_match = re.match(r'^blk\.(\d+)\.', tname)
        layer = int(layer_match.group(1)) if layer_match else None
        is_shared_expert = '_shexp.' in tname or '_chexp.' in tname
        is_routed_expert = '_exps.' in tname or '_chexps.' in tname or '.experts.' in tname
        is_expert = is_routed_expert or is_shared_expert
        is_expert_aux = bool(re.search(
            r'\.ffn_(gate_inp|gate_tid2eid|exp_probs_b)(?:\.|$)', tname))
        if is_expert:
            r['expert_bytes'] = r.get('expert_bytes', 0) + nbytes
            add_layer_bytes('_expert_layer_bytes', layer, nbytes)
            if is_shared_expert:
                # Shared experts ride with their layer's device: the `exps=CPU`
                # -ot catch-all does not match "shexp", so CPU-offloaded layers
                # still keep their shared expert on the owning GPU. Placement
                # needs this split to budget VRAM and RAM correctly.
                r['shexp_bytes'] = r.get('shexp_bytes', 0) + nbytes
                add_layer_bytes('_shexp_layer_bytes', layer, nbytes)
            else:
                add_layer_bytes('_routed_expert_layer_bytes', layer, nbytes)
        else:
            r['non_expert_bytes'] = r.get('non_expert_bytes', 0) + nbytes
            if is_expert_aux:
                # Routing metadata follows a whole expert-layer -ot pin. Keep
                # it in non_expert_bytes for the global file-size identity,
                # but expose the per-layer split so placement charges it to
                # the destination rather than both source and destination.
                r['expert_aux_bytes'] = r.get('expert_aux_bytes', 0) + nbytes
                add_layer_bytes('_expert_aux_layer_bytes', layer, nbytes)
            else:
                add_layer_bytes('_non_expert_layer_bytes', layer, nbytes)
            if tname == 'token_embd.weight' or tname == 'per_layer_token_embd.weight':
                # Input embeddings and qwen4exp's per-layer/n-gram PLE table stay
                # in host memory (llama.cpp never tensor-splits them onto CUDA).
                # Charging per_layer_token_embd (~27 GiB at Q3) as GPU backbone
                # makes every card look over-full and the model "not fit".
                r['token_embd_bytes'] = r.get('token_embd_bytes', 0) + nbytes
                if tname == 'per_layer_token_embd.weight':
                    r['has_per_layer_token_embd'] = 1
                    r['per_layer_token_embd_bytes'] = r.get('per_layer_token_embd_bytes', 0) + nbytes
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
    r: Dict[str, Any] = {
        'tensor_accounting_schema': 2,
        'has_per_layer_token_embd': 0,
        'fused': 0,
        'expert_bytes': 0,
        'non_expert_bytes': 0,
    }

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
    if r.get('ssm') or r.get('arch') in _RECURRENT_ARCHS:
        r['recurrent'] = 1
    if 'leading_dense' not in r and '_first_moe_layer' in r:
        r['leading_dense'] = r['_first_moe_layer']
        r['leading_dense_inferred'] = 1
    if 'expert_shared_count' not in r and '_inferred_expert_shared_count' in r:
        r['expert_shared_count'] = r['_inferred_expert_shared_count']
        r['expert_shared_count_inferred'] = 1
    layer_keys = (
        ('_expert_layer_bytes', 'expert_layer_bytes'),
        ('_routed_expert_layer_bytes', 'routed_expert_layer_bytes'),
        ('_shexp_layer_bytes', 'shexp_layer_bytes'),
        ('_expert_aux_layer_bytes', 'expert_aux_layer_bytes'),
        ('_non_expert_layer_bytes', 'non_expert_layer_bytes'),
    )
    layer_count = r.get('layers', 0)
    for private_key, public_key in layer_keys:
        values = r.pop(private_key, {})
        if values:
            count = max(layer_count, max(values) + 1)
            r[public_key] = [values.get(i, 0) for i in range(count)]
    r.pop('_align', None)
    # Private parser bookkeeping is not part of the public JSON contract.
    r.pop('_first_moe_layer', None)
    r.pop('_inferred_expert_shared_count', None)
    r.pop('_leading_dense_metadata', None)
    return r


# (metadata key, bash variable name, default-for-missing)
SHELL_KEY_MAP = [
    ('tensor_accounting_schema', 'TENSOR_ACCOUNTING_SCHEMA', 0),
    ('has_per_layer_token_embd', 'HAS_PER_LAYER_TOKEN_EMBD', 0),
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
    ('expert_aux_bytes',  'EXPERT_AUX_BYTES',    0),
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
