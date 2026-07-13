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
              embd=5120, ff=17408, ctx_train=262144, full_interval=4, ssm=True)
        r = parse(f.name)
    assert_eq(r['ssm'], 1, 'ssm')
    assert_eq(r['full_interval'], 4, 'full_interval')
    print('  ✓ ssm_hybrid')


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
    assert r == {'fused': 0, 'expert_bytes': 0, 'non_expert_bytes': 0}, r
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


def main():
    print('parse_gguf.py regression tests:')
    test_dense_llama()
    test_moe_qwen35()
    test_mla_deepseek()
    test_iswa_gemma()
    test_ssm_hybrid()
    test_tokenizer_hash_is_stable_and_vocab_sensitive()
    test_corrupted_gguf()
    test_shell_format_emits_all_keys()
    test_ik_llama_iq3_k_tensor_bytes()
    test_unknown_ttype_falls_back_to_4bpw()
    test_known_quant_table_has_ik_llama_ids()
    print('All tests passed.')


if __name__ == '__main__':
    sys.exit(main() or 0)
