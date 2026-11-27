#!/usr/bin/env python3
"""Regression tests for parse_gguf.py.

Run from the repo root:
    python3 tests/test_parse_gguf.py

Builds a handful of synthetic GGUFs covering the architectures that drive
distinct code paths (dense Llama-class, MoE, MLA/DeepSeek-class, ISWA, SSM
hybrid) and asserts the parser extracts the keys downstream code depends on.
No network, no model files, no build step — pure stdlib.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.path.join(ROOT, 'tests', 'build_synthetic_gguf.py')
PARSER = os.path.join(ROOT, 'tools', 'gguf', 'parse_gguf.py')


def build(out, **kwargs):
    cmd = ['python3', BUILDER, '--out', out]
    for k, v in kwargs.items():
        if v is True:
            cmd.append(f'--{k.replace("_", "-")}')
        elif v is False or v is None:
            continue
        elif isinstance(v, (list, tuple)):
            for item in v:
                cmd.append(f'--{k.replace("_", "-")}')
                cmd.append(str(item))
        else:
            cmd.append(f'--{k.replace("_", "-")}')
            cmd.append(str(v))
    subprocess.run(cmd, check=True, capture_output=True)


def parse(path):
    out = subprocess.run(['python3', PARSER, '--format', 'json', path],
                         check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def assert_eq(actual, expected, label):
    if actual != expected:
        raise AssertionError(f'{label}: expected {expected!r}, got {actual!r}')


def test_dense_llama():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='llama', name='Test-Llama-7B', layers=32,
              hkv=8, kl=128, vl=128, embd=4096, ff=14336, ctx_train=8192)
        r = parse(f.name)
    assert_eq(r['arch'], 'llama', 'arch')
    assert_eq(r['layers'], 32, 'layers')
    assert_eq(r['hkv'], 8, 'hkv')
    assert_eq(r['kl'], 128, 'kl')
    assert_eq(r['vl'], 128, 'vl')
    assert_eq(r['ctx_train'], 8192, 'ctx_train')
    assert_eq(r['name'], 'Test-Llama-7B', 'name')
    assert r.get('experts', 0) == 0, 'dense should have 0 experts'
    print('  ✓ dense_llama')


def test_moe_qwen35():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='qwen35moe', layers=40, hkv=2, kl=256, vl=256,
              embd=2048, experts=256, exp_used=8, exp_ff=512,
              ctx_train=262144, full_interval=4)
        r = parse(f.name)
    assert_eq(r['arch'], 'qwen35moe', 'arch')
    assert_eq(r['experts'], 256, 'experts')
    assert_eq(r['exp_used'], 8, 'exp_used')
    assert_eq(r['exp_ff'], 512, 'exp_ff')
    assert_eq(r['full_interval'], 4, 'full_interval')
    assert_eq(r['ctx_train'], 262144, 'ctx_train')
    print('  ✓ moe_qwen35')


def test_mla_deepseek():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='deepseek2', layers=61, hkv=128, kl=192, vl=128,
              kv_lora=512, q_lora=1536, embd=7168, ctx_train=163840)
        r = parse(f.name)
    assert_eq(r['kv_lora'], 512, 'kv_lora')
    assert_eq(r['q_lora'], 1536, 'q_lora')
    assert_eq(r['layers'], 61, 'layers')
    print('  ✓ mla_deepseek')


def test_iswa_gemma():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='gemma3', layers=42, hkv=4, kl=256, vl=256,
              swa=4096, embd=3840, ctx_train=131072)
        r = parse(f.name)
    assert_eq(r['swa'], 4096, 'swa')
    assert_eq(r['layers'], 42, 'layers')
    print('  ✓ iswa_gemma')


def test_ssm_hybrid():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='qwen35', layers=64, hkv=4, kl=256, vl=256,
              embd=5120, ff=17408, ctx_train=262144, full_interval=4, ssm=True,
              u32=['ssm.conv_kernel=4', 'ssm.group_count=16', 'ssm.inner_size=6144',
                   'ssm.time_step_rank=48'])
        r = parse(f.name)
    assert_eq(r['ssm'], 1, 'ssm')
    assert_eq(r['full_interval'], 4, 'full_interval')
    # Placement prices the per-slot recurrent state from these.
    for key, want in (('ssm_d_conv', 4), ('ssm_d_state', 128), ('ssm_n_group', 16),
                      ('ssm_d_inner', 6144), ('ssm_dt_rank', 48)):
        assert_eq(r.get(key), want, key)
    print('  ✓ ssm_hybrid')


def test_recurrent_state_beyond_ssm_state_size():
    """'recurrent' is the cache fact: state the backend cannot truncate, so a
    branch resumes only from a checkpoint. GLM-5.3-Flash (ssm.conv_kernel +
    kda.*) and Inkling (shortconv_kernel) state no ssm.state_size; treating
    them as plain attention restored 9 of 4,247 shared tokens on a branch."""
    cases = [
        (dict(arch='glm5next', layers=46, kv_lora=512,
              u32=['ssm.conv_kernel=4', 'kda.head_dim=128']), 0, 1),
        (dict(arch='inkling', swa=512, u32=['shortconv_kernel=4']), 0, 1),
        (dict(arch='future-linear', u32=['kda.head_dim=128']), 0, 1),
        (dict(arch='lfm2', layers=16), 0, 1),
        (dict(arch='qwen35', full_interval=4, ssm=True), 1, 1),
        (dict(arch='llama', layers=32, hkv=8), 0, 0),
    ]
    for kwargs, ssm, recurrent in cases:
        with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
            build(f.name, **kwargs)
            r = parse(f.name)
        assert_eq(r.get('ssm', 0), ssm, f"{kwargs['arch']} ssm")
        assert_eq(r.get('recurrent', 0), recurrent, f"{kwargs['arch']} recurrent")
    print('  ✓ recurrent_state_beyond_ssm_state_size')


def test_tokenizer_hash_is_stable_and_vocab_sensitive():
    with tempfile.NamedTemporaryFile(suffix='.gguf') as first, \
            tempfile.NamedTemporaryFile(suffix='.gguf') as same, \
            tempfile.NamedTemporaryFile(suffix='.gguf') as different:
        build(first.name, arch='qwen35', tokenizer_model='gpt2',
              tokenizer_pre='qwen35', vocab_size=64)
        build(same.name, arch='qwen35', tokenizer_model='gpt2',
              tokenizer_pre='qwen35', vocab_size=64)
        build(different.name, arch='qwen35', tokenizer_model='gpt2',
              tokenizer_pre='qwen35', vocab_size=65)
        first_hash = parse(first.name)['tokenizer_hash']
        same_hash = parse(same.name)['tokenizer_hash']
        different_hash = parse(different.name)['tokenizer_hash']
    assert len(first_hash) == 64, first_hash
    assert first_hash == same_hash, (first_hash, same_hash)
    assert first_hash != different_hash, (first_hash, different_hash)
    print('  ✓ tokenizer_hash_is_stable_and_vocab_sensitive')


def test_corrupted_gguf():
    """Non-GGUF input → empty dict, never crashes."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        f.write(b'NOT A GGUF FILE')
        f.flush()
        r = parse(f.name)
    assert r == {
        'tensor_accounting_schema': 3,
        'has_per_layer_token_embd': 0,
        'fused': 0,
        'expert_bytes': 0,
        'non_expert_bytes': 0,
    }, r
    print('  ✓ corrupted_gguf')


def test_shell_format_emits_all_keys():
    """Shell format must always emit every variable in SHELL_KEY_MAP, even
    when the GGUF is missing them — downstream bash relies on every var being
    set so `set -u` doesn't blow up."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='llama', layers=8)
        out = subprocess.run(['python3', PARSER, '--format', 'shell', f.name],
                             check=True, capture_output=True, text=True).stdout
    expected_vars = {
        'LAYER_COUNT', 'EXPERT_COUNT', 'HEAD_COUNT_KV', 'KEY_LENGTH',
        'VALUE_LENGTH', 'KEY_LENGTH_MLA', 'VALUE_LENGTH_MLA',
        'HAS_SSM', 'HAS_FUSED', 'EXPERT_BYTES',
        'NON_EXPERT_BYTES', 'MODEL_ARCH', 'EMBEDDING_LENGTH',
        'FEED_FORWARD_LENGTH', 'EXPERT_USED_COUNT', 'EXPERT_FF',
        'EXPERT_SHARED_FF', 'KV_LORA_RANK', 'Q_LORA_RANK', 'ROPE_DIM',
        'LEADING_DENSE', 'SLIDING_WINDOW', 'FULL_ATTN_INTERVAL', 'HAS_SHEXP',
        'CTX_TRAIN', 'GGUF_MODEL_NAME', 'GGUF_BASENAME', 'GGUF_QUANTIZED_BY',
        'GGUF_TOKENIZER_MODEL', 'GGUF_TOKENIZER_PRE', 'GGUF_VOCAB_SIZE',
        'GGUF_TOKENIZER_HASH',
    }
    emitted = {ln.split('=', 1)[0] for ln in out.splitlines() if '=' in ln}
    missing = expected_vars - emitted
    assert not missing, f'missing vars: {missing}'
    print('  ✓ shell_format_emits_all_keys')


def test_ik_llama_iq3_k_tensor_bytes():
    """ik_llama.cpp custom quants (type IDs 137+) must use their actual block
    sizes, not the F16 fallback. Issue #11: Kimi-K2.6-IQ3_K reported "313%
    expert ratio" because IQ3_K (type 138, 3.44 bpw) was treated as 16 bpw.

    Synthesizes one expert tensor with type 138 over 256k elements and
    verifies expert_bytes is ~110kB (256k × 110/256), not ~512kB (256k × 2)."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        # 256 * 1024 elements = 1024 IQ3_K blocks of 256 elements
        # Expected: 1024 × 110 = 112_640 bytes
        build(f.name, arch='qwen35moe', layers=1,
              tensor=['blk.0.ffn_down_exps.weight:262144:138'])
        r = parse(f.name)
    expected = 1024 * 110
    actual = r['expert_bytes']
    assert actual == expected, f'IQ3_K expert_bytes: expected {expected}, got {actual}'
    assert r['non_expert_bytes'] == 0, f'should not classify as non-expert: {r}'
    print('  ✓ ik_llama_iq3_k_tensor_bytes')


def test_unknown_ttype_falls_back_to_4bpw():
    """Unknown ttypes (e.g. brand-new ik_llama quant we haven't tabulated)
    must default to ~4 bpw (0.5 B/elem), not F16 (2 B/elem). The old fallback
    was the proximate cause of issue #11's expert-bytes over-count."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        # Type 999 is reserved-future; will hit the fallback.
        build(f.name, arch='qwen35moe', layers=1,
              tensor=['blk.0.ffn_down_exps.weight:1024:999'])
        r = parse(f.name)
    expected = 1024 // 2  # 0.5 B/elem fallback
    actual = r['expert_bytes']
    assert actual == expected, f'unknown ttype fallback: expected {expected}, got {actual}'
    print('  ✓ unknown_ttype_falls_back_to_4bpw')


def test_known_quant_table_has_ik_llama_ids():
    """Direct check that the parser's GGUF_TYPE_SIZE table covers the ik_llama
    custom quants we expect. Catches accidental deletion of these entries."""
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'gguf'))
    import parse_gguf  # noqa: E402
    sys.path.pop(0)
    required = [137, 138, 139, 140, 141]  # IQ2_K..IQ6_K
    for tid in required:
        assert tid in parse_gguf.GGUF_TYPE_SIZE, f'missing ik_llama ttype {tid}'
        bpb, epb = parse_gguf.GGUF_TYPE_SIZE[tid]
        assert epb == 256, f'ttype {tid}: epb should be 256, got {epb}'
        assert 60 < bpb < 220, f'ttype {tid}: bpb {bpb} out of plausible range'
    print('  ✓ known_quant_table_has_ik_llama_ids')


def test_per_layer_attention_arrays():
    """Mixed-head / explicit-window models state KV geometry only as per-layer
    arrays (MiMo-V2 style: 4 KV heads on full-attention blocks, 8 on windowed
    ones, NextN blocks at the end). The arrays must survive, a mixed head count
    must never collapse into the scalar, and later keys must stay aligned."""
    pattern = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0]
    heads = [4 if p == 0 else 8 for p in pattern]
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='mimo2', hkv_arr=','.join(map(str, heads)),
              swa_pattern=','.join(map(str, pattern)), layers=10, nextn=2,
              kl=192, vl=128, swa=128)
        r = parse(f.name)
    assert_eq(r.get('hkv_arr'), heads, 'hkv_arr')
    assert_eq(r.get('swa_pattern'), pattern, 'swa_pattern')
    assert 'hkv' not in r, f'mixed heads averaged into scalar hkv={r.get("hkv")}'
    assert_eq(r['layers'], 10, 'layers after arrays')
    assert_eq(r['nextn_predict_layers'], 2, 'nextn after arrays')
    assert_eq(r['swa'], 128, 'swa after arrays')
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='llama', hkv_arr='8,8,8,8', layers=4)
        r = parse(f.name)
    assert_eq(r.get('hkv'), 8, 'uniform array keeps the scalar contract')
    print('  ✓ per_layer_attention_arrays')


# Q8_0 (type 8): 34 bytes per 32 elements. A 256-wide row is 272 bytes.
ENGRAM_TABLE = 'blk.1.engram_embd.weight:256,1024:8'
ENGRAM_TABLE_BYTES = 1024 * 272
ENGRAM_DENSE = ['blk.0.attn_kv.weight:256,256:8', 'blk.1.attn_kv.weight:256,256:8',
                'blk.1.engram_wkv.weight:256,256:8', 'blk.1.ngram_proj.weight:256,256:8',
                'blk.1.ffn_gate_exps.weight:256,256,4:8']
DENSE_TENSOR_BYTES = 256 * 256 // 32 * 34


def build_engram(path, **kwargs):
    build(path, arch='deepseek41', layers=2, experts=4, exp_used=2,
          tensor=[ENGRAM_TABLE] + ENGRAM_DENSE, **kwargs)
    return parse(path)


def test_engram_tables_are_a_separate_sparse_lookup_tier():
    """DeepSeek-V4.1 Engram tables are gathered a few rows per token and the
    reviewed backend keeps them file-backed. The arch's own engram.layer_ids
    metadata, not the tensor name alone, identifies them. They stay in
    non_expert_bytes for the file-size identity but must leave the per-layer
    backbone charge, or placement bills ~195 GiB as GPU layer weights."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        r = build_engram(f.name, engram_layers='1')
        size = os.path.getsize(f.name)
        ranges = r.get('sparse_lookup_ranges') or []
        assert_eq(len(ranges), 1, 'one table range')
        assert_eq(ranges[0]['shard'], os.path.basename(f.name), 'range shard')
        assert_eq(ranges[0]['bytes'], ENGRAM_TABLE_BYTES, 'range bytes')
        off = ranges[0]['offset']
        # The header-only builder puts every tensor at data offset 0, so the
        # table starts where the aligned data section begins.
        assert_eq(off, (size + 31) // 32 * 32, 'range offset')
    assert_eq(r['tensor_accounting_schema'], 3, 'schema')
    # smalinin/llama.cpp 425badfe lists deepseek41 in llm_arch_is_hybrid.
    assert_eq(r.get('recurrent'), 1, 'deepseek41 recurrent/hybrid state')
    assert_eq(r.get('sparse_lookup_layer_ids'), [1], 'sparse_lookup_layer_ids')
    assert_eq(r.get('sparse_lookup_kind'), 'engram', 'sparse_lookup_kind')
    assert_eq(r.get('sparse_lookup_bytes'), ENGRAM_TABLE_BYTES, 'sparse_lookup_bytes')
    assert_eq(r.get('sparse_lookup_layer_bytes'), [0, ENGRAM_TABLE_BYTES], 'sparse_lookup_layer_bytes')
    # attn_kv + engram_wkv + an unrelated "ngram"-named projection stay ordinary.
    assert_eq(r['non_expert_layer_bytes'], [DENSE_TENSOR_BYTES, 3 * DENSE_TENSOR_BYTES], 'non_expert_layer_bytes')
    assert_eq(r['non_expert_bytes'], 4 * DENSE_TENSOR_BYTES + ENGRAM_TABLE_BYTES, 'non_expert_bytes identity')
    print('  ✓ engram_tables_are_a_separate_sparse_lookup_tier')


def test_engram_named_tensor_without_proof_stays_resident():
    """Negative cases: the same tensor without the arch's layer list, with the
    list under another arch prefix (an early mis-converted GGUF the backend does
    not read), or at an unlisted layer, remains ordinary resident backbone."""
    cases = (
        ('no metadata', {}),
        ('wrong prefix', {'engram_layers': '1', 'engram_prefix': 'deepseek4'}),
        ('unlisted layer', {'engram_layers': '0'}),
    )
    for label, extra in cases:
        with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
            r = build_engram(f.name, **extra)
        assert_eq(r.get('sparse_lookup_bytes', 0), 0, label + ': sparse_lookup_bytes')
        assert 'sparse_lookup_kind' not in r, f'{label}: unexpected sparse kind {r.get("sparse_lookup_kind")}'
        assert 'sparse_lookup_ranges' not in r, f'{label}: unexpected table ranges'
        assert_eq(r['non_expert_layer_bytes'][1], 3 * DENSE_TENSOR_BYTES + ENGRAM_TABLE_BYTES,
                  label + ': table charged to its layer')
    print('  ✓ engram_named_tensor_without_proof_stays_resident')


def test_lookup_metadata_on_an_architecture_without_adapter_stays_resident():
    """Negative: another architecture carrying the same metadata key and tensor
    name has no reviewed adapter, so its table stays an ordinary layer tensor."""
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='synthrow', layers=2, engram_layers='1',
              tensor=[ENGRAM_TABLE, 'blk.0.attn_kv.weight:256,256:8', 'blk.1.attn_kv.weight:256,256:8'])
        r = parse(f.name)
    assert_eq(r.get('sparse_lookup_bytes', 0), 0, 'sparse_lookup_bytes')
    assert 'sparse_lookup_layer_ids' not in r, 'unexpected layer ids'
    assert_eq(r['non_expert_layer_bytes'][1], DENSE_TENSOR_BYTES + ENGRAM_TABLE_BYTES, 'table charged to its layer')
    print('  ✓ lookup_metadata_on_an_architecture_without_adapter_stays_resident')


def test_synthetic_second_adapter_uses_its_own_metadata():
    """SYNTHETIC: a test-only adapter for a made-up dense architecture with a
    different metadata key and tensor name. It proves the adapter table, not the
    Engram spelling, drives classification. It is not a supported model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('parse_gguf_under_test', PARSER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SPARSE_LOOKUP_ADAPTERS['synthrow'] = {
        'kind': 'synthetic-rowtable', 'layer_key': 'rowtable.layer_ids', 'tensor': 'blk.{layer}.row_table.weight'}
    table = 'blk.1.row_table.weight:256,1024:8'
    dense = ['blk.0.attn_kv.weight:256,256:8', 'blk.1.attn_kv.weight:256,256:8',
             'blk.1.engram_embd.weight:256,256:8']  # the Engram name is ordinary here
    with tempfile.NamedTemporaryFile(suffix='.gguf') as f:
        build(f.name, arch='synthrow', layers=2, i32s=['rowtable.layer_ids=1'], tensor=[table] + dense)
        r = mod.parse(f.name)
        plain = parse(f.name)  # the shipped parser has no such adapter
    assert_eq(r.get('sparse_lookup_kind'), 'synthetic-rowtable', 'kind')
    assert_eq(r.get('sparse_lookup_bytes'), ENGRAM_TABLE_BYTES, 'bytes')
    assert_eq(r.get('sparse_lookup_layer_ids'), [1], 'layer ids')
    assert_eq(r['non_expert_layer_bytes'], [DENSE_TENSOR_BYTES, 2 * DENSE_TENSOR_BYTES], 'ordinary layer bytes')
    assert_eq(plain.get('sparse_lookup_bytes', 0), 0, 'shipped parser must not classify the synthetic table')
    print('  ✓ synthetic_second_adapter_uses_its_own_metadata')


def test_hostile_lengths_are_refused_without_huge_reads():
    """A header length larger than the file must be refused, not read: a
    2**60-byte key would otherwise make the parser allocate or spin."""
    import struct as st
    import tempfile
    import time as tm
    huge = 1 << 60
    def kv_header(tensor_count, kv_count):
        return b'GGUF' + st.pack('<I', 3) + st.pack('<Q', tensor_count) + st.pack('<Q', kv_count)
    def key(name):
        return st.pack('<Q', len(name)) + name
    cases = {
        'key length': kv_header(0, 1) + st.pack('<Q', huge),
        'string length': kv_header(0, 1) + key(b'general.name') + st.pack('<I', 8) + st.pack('<Q', huge),
        'array length': kv_header(0, 1) + key(b'tokenizer.ggml.scores') + st.pack('<I', 9) + st.pack('<I', 6) + st.pack('<Q', huge),
        'string array element': kv_header(0, 1) + key(b'tokenizer.ggml.tokens') + st.pack('<I', 9) + st.pack('<I', 8) + st.pack('<Q', 1) + st.pack('<Q', huge),
        'kv count': kv_header(0, huge),
        'tensor count': kv_header(huge, 0),
        'tensor name length': kv_header(1, 0) + st.pack('<Q', huge),
    }
    with tempfile.TemporaryDirectory() as d:
        for name, blob in cases.items():
            path = os.path.join(d, name.replace(' ', '_') + '.gguf')
            with open(path, 'wb') as fh:
                fh.write(blob + b'\0' * 64)
            start = tm.monotonic()
            r = parse(path)
            elapsed = tm.monotonic() - start
            assert elapsed < 2, '%s: took %.1fs' % (name, elapsed)
            assert r.get('tensor_accounting_schema') == 3, name
            assert 'name' not in r and 'tokenizer_hash' not in r, '%s: used a refused value: %r' % (name, r)
    # The refusal is the length bound itself, raised before any read.
    import importlib.util
    import io as _io
    spec = importlib.util.spec_from_file_location('parse_gguf_under_test', PARSER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for blob in (st.pack('<Q', huge) + b'\0' * 32, key(b'general.name') + st.pack('<I', 8) + st.pack('<Q', 1 << 20)):
        try:
            mod._read_kv(_io.BytesIO(blob), {}, 1)
        except ValueError as exc:
            assert 'exceeds the file' in str(exc), exc
        else:
            raise AssertionError('oversized length was read instead of refused')
    print('  ✓ hostile_lengths_are_refused_without_huge_reads')


def _types(r):
    return {t['type']: t for t in r.get('tensor_types', [])}


def test_type_histogram_counts_spans_and_names():
    """Per type id: table entries (empty ones too), elements, bytes by on-disk
    span and the distinct name patterns. Unknown ids are reported as numbers."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'hist.gguf')
        build(path, arch='llama', layers=2, real_spans=True, tensor=[
            'blk.0.attn_q.weight:256,32:8:8704',
            'zero.weight:0,5:8:0',
            'blk.1.attn_q.weight:256,32:8:8704',
            'blk.0.ffn_up_exps.weight:256,32,2:99:4096',
        ])
        r = parse(path)
    assert_eq(r['tensor_accounting_schema'], 3, 'accounting schema is unchanged')
    assert_eq(r['tensor_types'], [
        {'type': 8, 'tensors': 3, 'elements': 16384, 'bytes': 17408,
         'names': ['blk.*.attn_q.weight', 'zero.weight']},
        {'type': 99, 'tensors': 1, 'elements': 16384, 'bytes': 4096,
         'names': ['blk.*.ffn_up_exps.weight']},
    ], 'histogram')
    print('  ok type_histogram_counts_spans_and_names')


def test_type_histogram_colliding_id_differs_by_bytes():
    """The same id with the same element count is a different encoding when the
    spans differ: 2.25 and 1.3125 bits per weight here. Only spans tell them
    apart, which is why the histogram is never keyed by a type-size table."""
    with tempfile.TemporaryDirectory() as d:
        seen = []
        for nbytes in (4608, 2688):
            path = os.path.join(d, 'c%d.gguf' % nbytes)
            build(path, arch='llama', real_spans=True,
                  tensor=['blk.0.ffn_down_exps.weight:256,64:42:%d' % nbytes])
            seen.append(_types(parse(path))[42])
    assert_eq(seen[0]['elements'], seen[1]['elements'], 'same element count')
    assert_eq([t['bytes'] for t in seen], [4608, 2688], 'bytes differ for one id')
    print('  ok type_histogram_colliding_id_differs_by_bytes')


def test_type_histogram_merges_shards():
    with tempfile.TemporaryDirectory() as d:
        first = os.path.join(d, 'm-00001-of-00002.gguf')
        build(first, arch='llama', layers=2, real_spans=True,
              tensor=['blk.0.attn_q.weight:256,32:8:8704'])
        build(os.path.join(d, 'm-00002-of-00002.gguf'), arch='llama', real_spans=True,
              tensor=['blk.1.attn_q.weight:256,32:8:8704', 'blk.1.ffn_up_exps.weight:256,64:42:4608'])
        types = _types(parse(first))
        # A zero-filled shard 2 (a preallocated, interrupted download): the
        # histogram of shard 1 alone must not stand for the model.
        with open(os.path.join(d, 'm-00002-of-00002.gguf'), 'wb') as fh:
            fh.write(b'\0' * 4096)
        zeroed = parse(first)
        os.remove(os.path.join(d, 'm-00002-of-00002.gguf'))
        missing = parse(first)
    assert_eq(types[8]['tensors'], 2, 'tensors across shards')
    assert_eq(types[8]['bytes'], 2 * 8704, 'bytes across shards')
    assert_eq(types[42]['bytes'], 4608, 'second shard type')
    assert_eq(types[8]['names'], ['blk.*.attn_q.weight'], 'names merge')
    assert 'tensor_types' not in zeroed, 'a zero-filled shard must give no histogram: %r' % zeroed
    assert 'tensor_types' not in missing, 'a missing shard must give no histogram: %r' % missing
    print('  ok type_histogram_merges_shards')


def test_type_histogram_is_absent_when_offsets_cannot_be_trusted():
    spec = ['blk.%d.attn_q.weight:256,32:99:64' % i for i in range(4)]
    with tempfile.TemporaryDirectory() as d:
        # The default synthetic file has every tensor at offset 0 and no data.
        plain = os.path.join(d, 'plain.gguf')
        build(plain, arch='llama', tensor=[s.rsplit(':', 1)[0] for s in spec])
        assert 'tensor_types' not in parse(plain), 'zero-length synthetic data has no spans'

        good = os.path.join(d, 'good.gguf')
        build(good, arch='llama', real_spans=True, tensor=spec)
        assert_eq(_types(parse(good))[99]['tensors'], 4, 'good file')
        blob = open(good, 'rb').read()
        # Cut inside the last table entry: the table is truncated.
        cut_table = os.path.join(d, 'cut_table.gguf')
        with open(cut_table, 'wb') as fh:
            fh.write(blob[:len(blob) - 4 * 64 - 31 - 10])
        assert 'tensor_types' not in parse(cut_table), 'truncated table'
        # Cut into the data: the last tensor now starts past the end of the file.
        cut_data = os.path.join(d, 'cut_data.gguf')
        with open(cut_data, 'wb') as fh:
            fh.write(blob[:len(blob) - 100])
        assert 'tensor_types' not in parse(cut_data), 'interrupted download'
    print('  ok type_histogram_is_absent_when_offsets_cannot_be_trusted')


def test_type_histogram_is_absent_when_a_span_is_smaller_than_the_type_layout():
    """Q8_0 over 256x32 needs 8704 bytes; a 4000-byte span means a damaged file
    or another encoding, and the Go reader reports no histogram for it too."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'short.gguf')
        build(path, arch='llama', real_spans=True, tensor=['blk.0.attn_q.weight:256,32:8:4000'])
        assert 'tensor_types' not in parse(path), 'span smaller than the layout'
        ok = os.path.join(d, 'ok.gguf')
        build(ok, arch='llama', real_spans=True, tensor=['blk.0.attn_q.weight:256,32:8:8704'])
        assert_eq(_types(parse(ok))[8]['bytes'], 8704, 'span equal to the layout')
    print('  ok type_histogram_is_absent_when_a_span_is_smaller_than_the_type_layout')


def test_type_histogram_bounds_name_patterns():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'names.gguf')
        build(path, arch='llama', real_spans=True,
              tensor=['x%03d.weight:32:8:34' % i for i in range(130)])
        t = _types(parse(path))[8]
    assert_eq(t['tensors'], 130, 'every entry counted')
    assert_eq(len(t['names']), 128, 'names are capped')
    assert_eq(t['names'], sorted(t['names']), 'names are sorted')
    assert_eq(t['names_truncated'], True, 'cap is reported')
    print('  ok type_histogram_bounds_name_patterns')


def main():
    print('parse_gguf.py regression tests:')
    test_dense_llama()
    test_moe_qwen35()
    test_mla_deepseek()
    test_iswa_gemma()
    test_ssm_hybrid()
    test_recurrent_state_beyond_ssm_state_size()
    test_tokenizer_hash_is_stable_and_vocab_sensitive()
    test_corrupted_gguf()
    test_shell_format_emits_all_keys()
    test_ik_llama_iq3_k_tensor_bytes()
    test_unknown_ttype_falls_back_to_4bpw()
    test_known_quant_table_has_ik_llama_ids()
    test_per_layer_attention_arrays()
    test_engram_tables_are_a_separate_sparse_lookup_tier()
    test_engram_named_tensor_without_proof_stays_resident()
    test_lookup_metadata_on_an_architecture_without_adapter_stays_resident()
    test_synthetic_second_adapter_uses_its_own_metadata()
    test_hostile_lengths_are_refused_without_huge_reads()
    test_type_histogram_counts_spans_and_names()
    test_type_histogram_colliding_id_differs_by_bytes()
    test_type_histogram_merges_shards()
    test_type_histogram_is_absent_when_offsets_cannot_be_trusted()
    test_type_histogram_is_absent_when_a_span_is_smaller_than_the_type_layout()
    test_type_histogram_bounds_name_patterns()
    print('All tests passed.')


if __name__ == '__main__':
    sys.exit(main() or 0)
