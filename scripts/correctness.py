# type: ignore

import torch
import numpy as np
import argparse
import os

torch.set_float32_matmul_precision("highest")
torch.set_printoptions(precision=6, sci_mode=False)


def load_and_assert(fname, expected_shape):
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)
    data = np.fromfile(fname, dtype=np.float32)
    if data.size != np.prod(expected_shape):
        raise ValueError(
            f"Expected {np.prod(expected_shape)} "
            f"floats in {fname} but got {data.size}"
        )
    return torch.from_numpy(data.reshape(expected_shape)).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--embed", type=int, default=8)
    parser.add_argument("--q_bin", type=str, required=True, help="Path to Q.bin")
    parser.add_argument("--k_bin", type=str, required=True, help="Path to K.bin")
    parser.add_argument("--v_bin", type=str, required=True, help="Path to V.bin")
    parser.add_argument(
        "--o_bin", type=str, required=True, help="Path to O.bin (kernel output)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    bsz = args.bsz
    num_heads = args.num_heads
    seq_len = args.seq_len
    embed = args.embed
    device = args.device

    head_dim = embed // num_heads
    batch_heads = bsz * num_heads
    expected_elems = batch_heads * seq_len * head_dim

    # Load Q, K, V, O
    Qbh = load_and_assert(args.q_bin, (batch_heads, seq_len, head_dim))
    Kbh = load_and_assert(args.k_bin, (batch_heads, seq_len, head_dim))
    Vbh = load_and_assert(args.v_bin, (batch_heads, seq_len, head_dim))
    kernel_out = load_and_assert(args.o_bin, (batch_heads, seq_len, head_dim))

    # Reference attention: compute per head
    outputs = torch.empty(
        (batch_heads, seq_len, head_dim), device=device, dtype=torch.float32
    )
    for bh in range(batch_heads):
        q = Qbh[bh]
        k = Kbh[bh]
        v = Vbh[bh]
        scores = torch.matmul(q, k.transpose(0, 1)) / (head_dim**0.5)
        probs = torch.softmax(scores, dim=1)
        out = torch.matmul(probs, v)
        outputs[bh].copy_(out)

    # Compare
    diff = (kernel_out - outputs).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    print(f"\n\nMax abs diff = {max_err:.6e}, mean abs diff = {mean_err:.6e}")

    if max_err <= args.tolerance:
        print("\nPASS: kernel output matches PyTorch reference within tolerance.")
    else:
        print("\nFAIL: kernel differs from reference. Inspect slices below.")
        bad = (diff > args.tolerance).nonzero(as_tuple=False)
        print(f"\nNumber of bad elements: {bad.shape[0]}/{torch.numel(diff)}\n")
        for i in range(min(10, bad.shape[0])):
            b, h, r = bad[i].tolist()
            print(
                f"O[{b}, {h}, {r}]  ref={outputs[b,h,r].item():.6f}  "
                f"kernel={kernel_out[b,h,r].item():.6f}  "
                f"diff={diff[b,h,r].item():.6e}"
            )
    print()
