#!/usr/bin/env python3
"""Build a tiny synthetic GGUF file with chosen metadata, for tests.

The output file is a real GGUF that parse_gguf.py and llm-server can both read.
We skip real tensor data — tensor_count is set to whatever the caller passes,
and tensor headers are emitted but with zero-length data — so the file stays
under a few KB regardless of the model it represents.

Usage:
    python3 build_synthetic_gguf.py --arch llama --layers 32 --hkv 8 \
        --kl 128 --vl 128 --embd 4096 --ff 14336 --out /tmp/test.gguf
"""
import argparse
import struct
import sys


VT_UINT32 = 4
VT_STRING = 8
VT_UINT64 = 10


def w_uint32(buf, v):
    buf.append(struct.pack('<I', v))


def w_uint64(buf, v):
    buf.append(struct.pack('<Q', v))


def w_string(buf, s):
    b = s.encode('utf-8')
    w_uint64(buf, len(b))
    buf.append(b)


def w_kv_uint32(buf, key, val):
    w_string(buf, key)
    w_uint32(buf, VT_UINT32)
    w_uint32(buf, val)


def w_kv_string(buf, key, val):
    w_string(buf, key)
    w_uint32(buf, VT_STRING)
    w_string(buf, val)


def build(args):
    out = []
    out.append(b'GGUF')
    w_uint32(out, 3)               # version
    tensor_count = 0
    kv_pairs = []

    arch = args.arch
    if args.arch:
        kv_pairs.append(('general.architecture', 'string', args.arch))
    if args.name:
        kv_pairs.append(('general.name', 'string', args.name))
    if args.basename:
        kv_pairs.append(('general.basename', 'string', args.basename))
    if args.quantized_by:
        kv_pairs.append(('general.quantized_by', 'string', args.quantized_by))
    if args.layers is not None:
        kv_pairs.append((f'{arch}.block_count', 'uint32', args.layers))
    if args.hkv is not None:
        kv_pairs.append((f'{arch}.attention.head_count_kv', 'uint32', args.hkv))
    if args.kl is not None:
        kv_pairs.append((f'{arch}.attention.key_length', 'uint32', args.kl))
    if args.vl is not None:
        kv_pairs.append((f'{arch}.attention.value_length', 'uint32', args.vl))
    if args.embd is not None:
        kv_pairs.append((f'{arch}.embedding_length', 'uint32', args.embd))
    if args.ff is not None:
        kv_pairs.append((f'{arch}.feed_forward_length', 'uint32', args.ff))
    if args.experts is not None:
        kv_pairs.append((f'{arch}.expert_count', 'uint32', args.experts))
    if args.exp_used is not None:
        kv_pairs.append((f'{arch}.expert_used_count', 'uint32', args.exp_used))
    if args.exp_ff is not None:
        kv_pairs.append((f'{arch}.expert_feed_forward_length', 'uint32', args.exp_ff))
    if args.ctx_train is not None:
        kv_pairs.append((f'{arch}.context_length', 'uint32', args.ctx_train))
    if args.swa is not None:
        kv_pairs.append((f'{arch}.attention.sliding_window', 'uint32', args.swa))
    if args.full_interval is not None:
        kv_pairs.append((f'{arch}.attention.full_attention_interval', 'uint32', args.full_interval))
    if args.kv_lora is not None:
        kv_pairs.append((f'{arch}.attention.kv_lora_rank', 'uint32', args.kv_lora))
    if args.q_lora is not None:
        kv_pairs.append((f'{arch}.attention.q_lora_rank', 'uint32', args.q_lora))
    if args.ssm:
        kv_pairs.append((f'{arch}.ssm.state_size', 'uint32', 128))

    w_uint64(out, tensor_count)
    w_uint64(out, len(kv_pairs))
    for key, vtype, val in kv_pairs:
        if vtype == 'uint32':
            w_kv_uint32(out, key, val)
        elif vtype == 'string':
            w_kv_string(out, key, val)
        else:
            raise ValueError(f'unhandled type {vtype}')

    blob = b''.join(out)
    with open(args.out, 'wb') as f:
        f.write(blob)
    print(f'wrote {len(blob)} bytes to {args.out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--arch', default='llama')
    ap.add_argument('--name', default='')
    ap.add_argument('--basename', default='')
    ap.add_argument('--quantized-by', default='')
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--hkv', type=int, default=None)
    ap.add_argument('--kl', type=int, default=None)
    ap.add_argument('--vl', type=int, default=None)
    ap.add_argument('--embd', type=int, default=None)
    ap.add_argument('--ff', type=int, default=None)
    ap.add_argument('--experts', type=int, default=None)
    ap.add_argument('--exp-used', type=int, default=None)
    ap.add_argument('--exp-ff', type=int, default=None)
    ap.add_argument('--ctx-train', type=int, default=None)
    ap.add_argument('--swa', type=int, default=None)
    ap.add_argument('--full-interval', type=int, default=None)
    ap.add_argument('--kv-lora', type=int, default=None)
    ap.add_argument('--q-lora', type=int, default=None)
    ap.add_argument('--ssm', action='store_true')
    args = ap.parse_args()
    build(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
