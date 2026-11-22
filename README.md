<div align="center">

# Strata — Inkling-2-Mini Edition (AMD ROCm/HIP Native)

**High-throughput, long-context AMD ROCm/HIP + AVX2 inference for Inkling-2-Mini — built natively for RDNA4 (single & dual GPU).**

[![Repository](https://img.shields.io/badge/Repo-Tamanekokko%2FStrata-0ab887)](https://github.com/Tamanekokko/Strata)
[![Upstream](https://img.shields.io/badge/Fork%20of-kuraneko1%2FStrata-475569)](https://github.com/kuraneko1/Strata)
[![Engine](https://img.shields.io/badge/Engine-strata--inkling2-2563eb)](docs/INKLING2.md)
[![Verified GPU](https://img.shields.io/badge/Verified-AMD%20RX%209060%20XT%2016GB%20(1x%20%26%202x%20TP%3D2)-ed1c24)](#benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Highlights](#highlights) · [Benchmarks](#benchmarks) · [Architecture](#amd-rdna4-native-architecture) · [Build](#build-from-source) · [Quick Start](#quick-start)

</div>

---

**`Tamanekokko/Strata`** is an **AMD ROCm/HIP-native** rewrite of the `strata-inkling2` inference path forked from [kuraneko1/Strata](https://github.com/kuraneko1/Strata) (upstream [Niko1221/Strata](https://github.com/Niko1221/Strata)).

By dropping NVIDIA/CUDA-compatibility constraints and targeting **AMD RDNA4 (`gfx1200`)** and PCIe P2P directly, dual **RX 9060 XT 16 GB** decode throughput jumps from **`46.3 t/s` → `73.8 t/s` (+59.4%)** across **65K–1M context** with **zero CPU expert misses** and **100% greedy token parity**.

## Highlights

- **Full TP=2 across every layer (`--peer-device 1`)** — Attention (Q/K/V/Gate/Rel/O-proj + KV cache), MoE (expert-internal `n_ff = 1152 → 576 + 576` across all 128 experts, zero routing skew), and LM Head (`201k → 100.5k + 100.5k` vocab) are all split 50/50 across GPU0 and GPU1.
- **Push-only FineGrained P2P spin-barrier + full 48-layer Dual `hipGraph`** — Eliminates 96+ cross-GPU `hipEvent` syncs and `hipMemcpyPeerAsync` calls per token. GPU0 and GPU1 each launch a single independent `hipGraphExec_t` covering all 48 virtual layers + LM Head, synchronizing directly over PCIe 5.0 P2P via `hipDeviceMallocFinegrained` posted writes and `__hip_atomic_load/store(..., __HIP_MEMORY_SCOPE_SYSTEM)` (~13 µs round-trip).
- **Zero-Group 2-Kernel T=1 NVFP4 Fused MoE** — Dedicated RDNA4 `native_gu_nvfp4_72_situ_q8_1_t1_kernel` + `native_down_nvfp4_18_combine_t1_kernel` (144 blocks × 128 threads, SMEM `s_ue4m3` LUT, zero `k_group_experts` / zero `k_combine` HBM round-trip).
- **RDNA4 wave32 kernel fusions & FP16 loop controllers** — Fused `k_qkv_prep_rel_t1` (3→1 kernels), fused `k_p2p_wait_add_shortconv_t1`, inline RMS norm in `k_mix_normed_fused_1pass`, inline P2P push in `k_rms_norm_q8_1` / `k_rms_norm_3_q8_1` / `k_route_inkling2`, hardware `v_exp_f32` SWA attention, and FP16-converted 3-way loop controllers (340 MB → 170 MB/tok).
- **Up to 1M context on 2× 16 GB VRAM** — Halving KV cache per GPU leaves `3.5–7.4 GiB` free VRAM per GPU up to 524K upfront and supports `1,048,576` context at `73.7 t/s` with 100% of experts resident in GPU VRAM.

## Benchmarks

Measured on real hardware with `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--bench --kv-q8`, 4,096-token prompt, 128 generated tokens, DSpark OFF, default settings) on **AMD Radeon RX 9060 XT 16 GB (`gfx1200`, Ryzen 5 7600, ROCm 10.1.0)**:

### Dual GPU — 2× RX 9060 XT 16 GB (`HIP_VISIBLE_DEVICES=0,1 --peer-device 1`)

| `--max-context` | Prefill (4K tok) | Decode (128 tok) | vs Prior ROCm Port | Free VRAM (GPU0 / GPU1) | CPU Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`65536`** (`full_vram`, upfront KV) | **`2,465.1 t/s`** | **`73.80 t/s`** | **`+59.4%`** (`46.3 → 73.8`) | `7,142 / 7,406 MiB` | `0` |
| **`131072`** (`full_vram`, upfront KV) | **`2,467.1 t/s`** | **`73.77 t/s`** | **`+59.3%`** (`46.3 → 73.8`) | `6,622 / 6,886 MiB` | `0` |
| **`262144`** (`full_vram`, upfront KV) | **`2,470.0 t/s`** | **`73.82 t/s`** | **`+59.8%`** (`46.2 → 73.8`) | `5,582 / 5,846 MiB` | `0` |
| **`524288`** (`full_vram`, upfront KV) | **`2,468.1 t/s`** | **`73.74 t/s`** | **`+63.1%`** (`45.2 → 73.7`) | `3,502 / 3,766 MiB` | `0` |
| **`1048576`** (1M, dynamic KV) | **`2,456.5 t/s`** | **`73.70 t/s`** | **New (1M Full-GPU)** | `7,404 / 7,666 MiB` | `0` |

### Single GPU — 1× RX 9060 XT 16 GB (`HIP_VISIBLE_DEVICES=0`)

| `--max-context` | Prefill (4K tok) | Decode (64 tok) | vs Prior ROCm Port | VRAM Experts |
| --- | ---: | ---: | ---: | ---: |
| **`65536`** (`core_resident_hybrid`) | **`2,093.0 t/s`** | **`49.23 t/s`** | **`+11.4%`** (`44.2 → 49.2`) | `2,746 / 3,072` (`tier1` 100%) |

## AMD RDNA4 Native Architecture

| Component | Before (CUDA-port HIP) | After (AMD RDNA4 Native) |
| --- | --- | --- |
| **Attention & KV Cache** | GPU0 only (`100%` weights & KV) | **TP=2 50/50** (12+12 Q heads, 4+4 KV heads, halved KV/GPU) |
| **MoE Placement** | EP `64 + 64` experts (routing skew 5:3 / 6:2) | **TP=2 `576 + 576` inside all 128 experts** (exact 50/50) |
| **MoE T=1 NVFP4 Kernel** | `k_group` + 3-pass `native_expert_grouped` + `k_combine` | **Zero-Group 2-Kernel Fused** (`gu_situ_q8_1_t1` + `down_combine_t1`) |
| **Cross-GPU Sync** | 96× `hipEvent` + 192× `hipMemcpyPeerAsync` (~4.5 ms/tok) | **Push-only FineGrained P2P + Local Spin + Direct-to-Peer Output** |
| **Decode Graph** | Disabled on dual-GPU | **Full 48-layer + LM Head Dual `hipGraph`** (`tp_graph0` + `tp_graph1`) |
| **LM Head (`201k` Q8_0)** | GPU0 only (`1.90 ms`) | **TP=2 `100.5k + 100.5k`** (`0.88 ms`) + on-GPU argmax merge |
| **Loop Controllers** | F32 (`340 MB/tok`), 6 kernels/layer | **FP16 (`170 MB/tok`), 3-way fused** (2 kernels/layer) |
| **QKV Prep & Norm/Mix** | Separate `rms_norm`, `shortconv`, `clip_tau`, `rel_table` | Inline RMS norm + `k_qkv_prep_rel_t1` + `k_p2p_wait_add_shortconv_t1` |

## Build from Source

Requires **Linux x86_64** (AVX2 CPU), **CMake 3.24+**, a **C++20** compiler, and **ROCm / HIP** (tested on ROCm 10.1.0 with `gfx1200`).

```bash
git clone https://github.com/Tamanekokko/Strata.git
cd Strata

# Set your GPU architecture (gfx1200 = RX 9060 XT, gfx1201 = RX 9070/XT, gfx1100 = RX 7900 XTX)
cmake -S engine -B engine/build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP=ON \
  -DSTRATA_HIP_ARCH=gfx1200 \
  -DSTRATA_BUILD_INKLING2=ON \
  -DSTRATA_BUILD_SERVER=ON

cmake --build engine/build-hip --target strata-inkling2 strata -j$(nproc)
```

## Quick Start

### Dual-GPU Benchmark (2× RX 9060 XT, Full TP=2 + Dual `hipGraph`)

```bash
HIP_VISIBLE_DEVICES=0,1 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --peer-device 1 --max-context 262144 --kv-q8 \
  --bench --prompt "Hello" --max-new 128
```

### Single-GPU Benchmark (1× RX 9060 XT)

```bash
HIP_VISIBLE_DEVICES=0 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 65536 --kv-q8 \
  --bench --prompt "Hello" --max-new 128
```

### Runtime Diagnostic Flags

- `STRATA_NO_TP_GRAPH=1` — Disable Dual `hipGraph` (run Push-only P2P on direct streams)
- `STRATA_NO_P2P_SPIN=1` — Fall back from FineGrained P2P spin-barrier to `hipEvent` + `hipMemcpyPeerAsync`
- `STRATA_NO_TP_MOE=1` — Fall back from expert-internal TP=2 (`576+576`) to 64+64 EP split
- `STRATA_NO_CTRL_F16=1` — Keep loop controllers in F32 instead of FP16
- `STRATA_EV_PROF=1` — Print zero-host-sync per-stage GPU timing breakdown (with `STRATA_NO_TP_GRAPH=1`)

## License

MIT — see [LICENSE](LICENSE). Upstream credits: [Niko1221/Strata](https://github.com/Niko1221/Strata) and [kuraneko1/Strata](https://github.com/kuraneko1/Strata).
