<div align="center">

# Strata — Inkling-2-Mini Edition (AMD ROCm/HIP Native + QSA)

**High-throughput, long-context AMD ROCm/HIP + AVX2 inference for Inkling-2-Mini — built natively for RDNA4 (single & dual GPU) with QSA (Qwen Sparse Attention).**

[![Repository](https://img.shields.io/badge/Repo-Tamanekokko%2FStrata-0ab887)](https://github.com/Tamanekokko/Strata)
[![Upstream](https://img.shields.io/badge/Fork%20of-kuraneko1%2FStrata-475569)](https://github.com/kuraneko1/Strata)
[![QSA Indexer](https://img.shields.io/badge/HuggingFace-QSA%20Indexer%20v3-ffcc4d)](https://huggingface.co/tamanekokoro/Inkling-2-Mini-QSA-Indexer)
[![Engine](https://img.shields.io/badge/Engine-strata--inkling2-2563eb)](docs/INKLING2.md)
[![Verified GPU](https://img.shields.io/badge/Verified-AMD%20RX%209060%20XT%2016GB%20%281x%20%26%202x%20TP%3D2%29-ed1c24)](#benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Highlights](#highlights) · [QSA Sparse Attention](#qsa-qwen-sparse-attention) · [Benchmarks](#benchmarks) · [Architecture](#amd-rdna4-native-architecture) · [Build](#build-from-source) · [Quick Start](#quick-start)

</div>

---

**`Tamanekokko/Strata`** is an **AMD ROCm/HIP-native** rewrite of the `strata-inkling2` inference path forked from [kuraneko1/Strata](https://github.com/kuraneko1/Strata) (upstream [Niko1221/Strata](https://github.com/Niko1221/Strata)), extended with **QSA (Qwen Sparse Attention)** block-sparse full-attention acceleration for long contexts on both single-GPU and dual-GPU AMD Radeon systems.

By targeting **AMD RDNA4 (`gfx1200`)** and PCIe P2P directly, dual **RX 9060 XT 16 GB** dense decode throughput reaches **`81.6 t/s`** (`--bench`, 65K/256K ctx), **`80.6 t/s`** (short prompt, 65K), and **`76.6 t/s`** (4K full-SWA prompt, 65K) with **zero CPU expert misses**, while supporting up to **`--max-context` 524K** (upfront KV) and **`1M`** (dynamic KV). On a single **RX 9060 XT 16 GB** (`core_resident_hybrid`), dense decode reaches **`53.0 t/s`** (`--bench` / short prompt, 65K) and **`51.2 t/s`** (4K full-SWA prompt, 65K). With **QSA (`--qsa-indexer`)** enabled on real long-context prompts, 65K prefill speeds up by **+53% on 1× RX 9060 XT** (`1,057 → 1,614 t/s`) and **+32% on 2× RX 9060 XT** (`1,541 → 2,040 t/s`), while 256K NIAH runs at **`1,844 t/s` prefill / `61.1 t/s` decode** with exact needle recall.

## Highlights

- **QSA (Qwen Sparse Attention) for Full-Attention Layers (`--qsa-indexer`)** — Block-sparse top-K KV selection (`blk_sz=4` tokens/block) across all 8 full-attention virtual layers (`vl = 5, 11, 17, 23, 29, 35, 41, 47`) for both single-GPU and dual-GPU TP=2, plus teacher-dump (`--dump-qsa-teacher`, `--dump-qsa-manifest`) and Modal A100 distillation script ([`tools/qsa/train_qsa_modal_v3.py`](tools/qsa/train_qsa_modal_v3.py)). Pretrained v3 indexer weights (22.5 MiB) are hosted at [`tamanekokoro/Inkling-2-Mini-QSA-Indexer`](https://huggingface.co/tamanekokoro/Inkling-2-Mini-QSA-Indexer).
- **Single-GPU AMD Radeon Support & HIP Prefill Fix** — Works out-of-the-box on a single 16 GB Radeon (`HIP_VISIBLE_DEVICES=0`) using `core_resident_hybrid`, with RDNA4 WMMA Q8 prefill (`k_attn_q8_gqa3_pp_wmma`) set as the default HIP prefill attention kernel.
- **Full TP=2 across every layer (`--peer-device 1`)** — Attention (Q/K/V/Gate/Rel/O-proj + KV cache), MoE (expert-internal `n_ff = 1152 → 576 + 576` across all 128 experts, zero routing skew), and LM Head (`201k → 100.5k + 100.5k` vocab) are all split 50/50 across GPU0 and GPU1.
- **Push-only FineGrained P2P spin-barrier + full 48-layer Dual `hipGraph`** — Eliminates 96+ cross-GPU `hipEvent` syncs and `hipMemcpyPeerAsync` calls per token. GPU0 and GPU1 each launch a single independent `hipGraphExec_t` covering all 48 virtual layers + LM Head, synchronizing directly over PCIe 5.0 P2P via `hipDeviceMallocFinegrained` posted writes and `__hip_atomic_load/store(..., __HIP_MEMORY_SCOPE_SYSTEM)` (~13 µs round-trip).
- **Zero-Group 2-Kernel T=1 NVFP4 Fused MoE (1-GPU & 2-GPU)** — Dedicated RDNA4 `native_gu_nvfp4_72_situ_q8_1_t1_kernel` + `native_down_nvfp4_18_combine_t1_kernel` (2-GPU TP=2, `n_ff=576`) / `native_down_nvfp4_36_combine_t1_kernel` (1-GPU 36 virtual core layers, `n_ff=1152`) with SMEM `s_ue4m3` LUT and zero `k_group_experts` / `k_combine` HBM round-trip.
- **RDNA4 wave32 dense kernel fusions** — 512-thread 1-pass SWA attention (`k_attn_q8_gqa3_swa_1pass_256`, 32 half-warps, `__half` LDS) on both 1-GPU and 2-GPU, fused Mix+RMSNorm+Q8_1+P2P (`k_mix_rms_norm_q8_1_t1`), fused Router GEMV+Top-K (`k_gemv_route_inkling2_t1`), 3-way FP16 loop controllers (`uint4` 128-bit load, 340 MB → 170 MB/tok), fused `k_qkv_prep_rel_t1`, fused `k_p2p_wait_add_shortconv_t1`, inline P2P push/signal in `k_rms_norm_q8_1` / `k_rms_norm_3_q8_1`, and `PAIR=true` K+V MMVQ.
- **Up to 524K upfront KV (and 1M dynamic-KV config) on 2× 16 GB VRAM** — Halving KV cache per GPU leaves `3.5–7.4 GiB` free VRAM per GPU up to `524,288` upfront KV with 100% of experts resident in GPU VRAM, and supports `--max-context 1048576` with dynamic KV allocation (`73.7 t/s` on 4K-prompt benchmark; full 1M KV fill exceeds 2×16 GB VRAM).

## QSA (Qwen Sparse Attention)

Inkling-2-Mini alternates 5 sliding-window attention (SWA) layers with 1 full-attention layer (8 full-attention layers out of 48 virtual layers). At 65K–256K context, full-attention KV reads dominate prefill and long-context decode.

`strata-inkling2` implements a lightweight learned **QSA block indexer** (`h_idx=4`, `d_idx=128`, block size `4` tokens, NoPE `rope_theta=0`, 22.5 MiB total across 8 layers) that scores every 4-token KV block and selects the top-K remote blocks alongside always-kept `--qsa-sink` initial tokens and `--qsa-local` recent tokens, feeding directly into the WMMA sparse prefill and GQA3 sparse decode kernels.

### Long-Context Benchmarks (Dense vs QSA Indexer v3)

Measured on **AMD Radeon RX 9060 XT 16 GB (`gfx1200`, ROCm 10.1.0)** with `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--kv-q8`), QSA settings `--qsa-budget 4096 --qsa-local 256 --qsa-sink 16` (NIAH rows are single-needle, single-prompt probes):

| Task & Context | Config | Mode | Prefill | Decode | Accuracy / Notes |
| --- | --- | --- | ---: | ---: | --- |
| **65K NIAH** (needle `749281`) | 1× RX 9060 XT | Dense | `1,057 t/s` | `38.4 t/s` | Exact match (`749281`) |
| **65K NIAH** (needle `749281`) | 1× RX 9060 XT | **QSA v3** | **`1,614 t/s` (+53%)** | **`43.8 t/s` (+14%)** | **Exact match (`749281`)** |
| **65K NIAH** (needle `749281`) | 2× RX 9060 XT (TP=2) | Dense | `1,541 t/s` | `62.6 t/s` | Exact match (`749281`) |
| **65K NIAH** (needle `749281`) | 2× RX 9060 XT (TP=2) | **QSA v3** | **`2,040 t/s` (+32%)** | **`66.4 t/s` (+6%)** | **Exact match (`749281`)** |
| **256K NIAH** (`261,888 tok`, needle `831942`) | 2× RX 9060 XT (TP=2) | Dense | `821.4 t/s` (`318.8 s`) | `46.2 t/s` | Exact match (`831942`) |
| **256K NIAH** (`261,888 tok`, needle `831942`) | 2× RX 9060 XT (TP=2) | **QSA v3** | **`1,844 t/s` (+124%, `142.0 s`)** | **`61.1 t/s` (+32%)** | **Exact match (`831942`)** |
| **32K EN/JP Summarization** | 2× RX 9060 XT (TP=2) | Dense | `1,679 t/s` | `68.5 t/s` | Baseline bullet summary |
| **32K EN/JP Summarization** | 2× RX 9060 XT (TP=2) | **QSA v3** | **`1,883 t/s` (+12%)** | **`68.3 t/s` (~0%)** | Coherent bullet summary |

> **Current limitations of QSA Indexer v3:** Single-needle retrieval (65K / 256K) and long-document summarization work reliably, while dense multi-needle extraction (e.g., 3 separate keys at 65K) and verbatim multi-identifier code recall can still miss secondary needles or drift compared to exact dense attention. For workloads requiring bit-exact dense attention, omit `--qsa-indexer`.

## Benchmarks

### Latest Dense Kernel-Fusion Throughput (`--kv-q8`, greedy; 65K ctx unless noted)

Measured on **AMD Radeon RX 9060 XT 16 GB (`gfx1200`, Ryzen 5 7600, ROCm 10.1.0)** with `Inkling-2-Mini-UD-Q4_K_M.gguf` (DSpark OFF, Dense attention). `--bench` uses the default 2-token prompt; short and 4K full-SWA rows use real tokenized prompts via `--tokens-file`. `CPU Misses` is the cumulative `pp+tg` count (on 1-GPU `core_resident_hybrid`, non-resident outer-layer experts stream from host RAM):

| Config | Workload | Before Fusions (`6697fd5`) | After Fusions | Gain | CPU Misses (`pp+tg`) |
| --- | --- | ---: | ---: | ---: | ---: |
| **2× RX 9060 XT (TP=2)** | `--bench` (2 + 128 tok, 65K / 256K ctx) | — | **`81.60 t/s`** / **`81.58 t/s`** | — | `0` |
| **2× RX 9060 XT (TP=2)** | Short prompt (`--tokens-file`, 21 + 256 tok) | `74.55 t/s` | **`80.56 t/s`** | **`+8.1%`** | `0` |
| **2× RX 9060 XT (TP=2)** | 4K full-SWA prompt (`--tokens-file`, 4,096 + 128 tok) | `68.11 t/s` | **`76.59 t/s`** | **`+12.5%`** | `0` |
| **1× RX 9060 XT** | `--bench` (2 + 64 tok, 65K ctx) | — | **`53.01 t/s`** | — | `722` |
| **1× RX 9060 XT** | Short prompt (`--tokens-file`, 21 + 256 tok) | `50.87 t/s` | **`52.98 t/s`** | **`+4.1%`** | `2,290` |
| **1× RX 9060 XT** | 4K full-SWA prompt (`--tokens-file`, 4,096 + 128 tok) | — | **`51.19 t/s`** | — | `64,005` |

### Context-Scaling Baseline (RDNA4 Native TP=2 vs Prior ROCm Port, commit `9d1e76a`)

Measured prior to the latest kernel-fusion pass using `Inkling-2-Mini-UD-Q4_K_M.gguf` (`--kv-q8`, DSpark OFF, Dense attention, synthetic 4K prefill + short-window decode sweep) on **AMD Radeon RX 9060 XT 16 GB (`gfx1200`, Ryzen 5 7600, ROCm 10.1.0)**:

#### Dual GPU — 2× RX 9060 XT 16 GB (`HIP_VISIBLE_DEVICES=0,1 --peer-device 1`)

| `--max-context` | Prefill (4K tok) | Decode (128 tok) | vs Prior ROCm Port | Free VRAM (GPU0 / GPU1) | CPU Misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| **`65536`** (`full_vram`, upfront KV) | **`2,465.1 t/s`** | **`73.80 t/s`** | **`+59.4%`** (`46.3 → 73.8`) | `7,142 / 7,406 MiB` | `0` |
| **`131072`** (`full_vram`, upfront KV) | **`2,467.1 t/s`** | **`73.77 t/s`** | **`+59.3%`** (`46.3 → 73.8`) | `6,622 / 6,886 MiB` | `0` |
| **`262144`** (`full_vram`, upfront KV) | **`2,470.0 t/s`** | **`73.82 t/s`** | **`+59.8%`** (`46.2 → 73.8`) | `5,582 / 5,846 MiB` | `0` |
| **`524288`** (`full_vram`, upfront KV) | **`2,468.1 t/s`** | **`73.74 t/s`** | **`+63.1%`** (`45.2 → 73.7`) | `3,502 / 3,766 MiB` | `0` |
| **`1048576`** (1M, dynamic KV) | **`2,456.5 t/s`** | **`73.70 t/s`** | **New (1M dynamic KV)** | `7,404 / 7,666 MiB` | `0` |

#### Single GPU — 1× RX 9060 XT 16 GB (`HIP_VISIBLE_DEVICES=0`)

| `--max-context` | Prefill (4K tok) | Decode (64 tok) | vs Prior ROCm Port | VRAM Experts |
| --- | ---: | ---: | ---: | ---: |
| **`65536`** (`core_resident_hybrid`) | **`2,093.0 t/s`** | **`49.23 t/s`** | **`+11.4%`** (`44.2 → 49.2`) | `2,746 / 3,072` (`tier1` 100%) |

## AMD RDNA4 Native Architecture

| Component | Before (CUDA-port HIP) | After (AMD RDNA4 Native) |
| --- | --- | --- |
| **Attention & KV Cache** | GPU0 only (`100%` weights & KV) | **TP=2 50/50** (12+12 Q heads, 4+4 KV heads, halved KV/GPU) |
| **Long-Context Full Attention** | Exact dense scan only | **Optional QSA Block-Sparse Indexer** (`--qsa-indexer`, 4-tok blocks, WMMA PP + GQA3 TG) |
| **MoE Placement** | EP `64 + 64` experts (routing skew 5:3 / 6:2) | **TP=2 `576 + 576` inside all 128 experts** (exact 50/50) |
| **SWA Attention (40 layers)** | 2-pass split-K (`k_attn_split` + `k_attn_reduce`) | **512-thread 1-Pass SWA** (`k_attn_q8_gqa3_swa_1pass_256`, 32 half-warps, `__half` LDS, 1-GPU & 2-GPU) |
| **MoE T=1 NVFP4 Kernel** | `k_group` + 3-pass `native_expert_grouped` + `k_combine` | **Zero-Group 2-Kernel Fused** (`gu_situ_q8_1_t1` + `down_18_combine_t1` / `down_36_combine_t1`) |
| **MoE Router** | Separate `gemv` + `k_route_inkling2` | **Fused `k_gemv_route_inkling2_t1`** (16 blocks × 8 warps, LDS-shared `x`) |
| **Cross-GPU Sync** | 96× `hipEvent` + 192× `hipMemcpyPeerAsync` (~4.5 ms/tok) | **Push-only FineGrained P2P + Inline Signal (`k_rms_norm_3_q8_1` / `k_mix_rms_norm_q8_1_t1`)** |
| **Decode Graph** | Disabled on dual-GPU | **Full 48-layer + LM Head Dual `hipGraph`** (`tp_graph0` + `tp_graph1`) |
| **LM Head (`201k` Q8_0)** | GPU0 only (`1.90 ms`) | **TP=2 `100.5k + 100.5k`** (`k_mmvq_q8_0_smallT` on both GPUs + on-GPU argmax merge) |
| **Loop Controllers** | F32 (`340 MB/tok`), 6 kernels/layer | **FP16 (`170 MB/tok`), 3-way fused** (`uint4` load, 2 kernels/layer) |
| **QKV Prep & Norm/Mix** | Separate `mix`, `rms_norm`, `q8_1`, `shortconv`, `rel_table` | `k_mix_rms_norm_q8_1_t1` + `k_qkv_prep_rel_t1` + `k_p2p_wait_add_shortconv_t1` |

## Build from Source

Requires **Linux x86_64** (AVX2 CPU), **CMake 3.24+**, **Ninja**, a **C++20** compiler, and **ROCm / HIP** (tested on ROCm 10.1.0 with `gfx1200`).

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

### Single-GPU with QSA (1× RX 9060 XT, Long Context)

Download the pretrained QSA Indexer v3 (`22.5 MiB`) from Hugging Face:

```bash
curl -L -o inkling2-qsa-indexer-v3.bin \
  https://huggingface.co/tamanekokoro/Inkling-2-Mini-QSA-Indexer/resolve/main/inkling2-qsa-indexer-v3.bin

HIP_VISIBLE_DEVICES=0 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 66560 --kv-q8 \
  --qsa-indexer ./inkling2-qsa-indexer-v3.bin \
  --qsa-budget 4096 --qsa-local 256 --qsa-sink 16 \
  --prompt "Hello" --max-new 128
```

### Dual-GPU with QSA (2× RX 9060 XT, TP=2 + 256K Context)

```bash
HIP_VISIBLE_DEVICES=0,1 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --peer-device 1 --max-context 262144 --kv-q8 \
  --qsa-indexer ./inkling2-qsa-indexer-v3.bin \
  --qsa-budget 4096 --qsa-local 256 --qsa-sink 16 \
  --prompt "Hello" --max-new 128
```

### Dense Benchmark (without QSA, built-in `--bench`)

```bash
# Dual-GPU (2x RX 9060 XT)
HIP_VISIBLE_DEVICES=0,1 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --peer-device 1 --max-context 262144 --kv-q8 \
  --bench --max-new 128

# Single-GPU (1x RX 9060 XT)
HIP_VISIBLE_DEVICES=0 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 65536 --kv-q8 \
  --bench --max-new 64
```

### Distilling Your Own QSA Indexer

1. Collect dense full-attention teacher targets (`.qsat` files) from a manifest of tokenized `.tok` sequences:

```bash
HIP_VISIBLE_DEVICES=0 ./engine/build-hip/strata-inkling2 \
  --model /path/to/Inkling-2-Mini-UD-Q4_K_M.gguf \
  --max-context 16384 --kv-q8 --chunk 2048 \
  --dump-qsa-teacher ./teacher_v3 \
  --dump-qsa-manifest ./corpus_v3/manifest.txt
```

2. Upload `teacher_v3/` to the Modal Volume (`inkling2-qsa-data`) at `/teacher_v3` and run [`tools/qsa/train_qsa_modal_v3.py`](tools/qsa/train_qsa_modal_v3.py) on an A100 GPU to distill a `QSAW` binary indexer:

```bash
modal volume create inkling2-qsa-data
modal volume put inkling2-qsa-data ./teacher_v3 /teacher_v3
modal run tools/qsa/train_qsa_modal_v3.py
modal volume get inkling2-qsa-data /inkling2-qsa-indexer-v3.bin ./inkling2-qsa-indexer-v3.bin
```

### CLI & Runtime Diagnostic Flags

- `--qsa-indexer <path>` — Load a `QSAW` QSA indexer binary (enables block-sparse full attention)
- `--qsa-budget <int>` — Total KV token budget per full-attention layer (default: `2048`, recommended: `4096`)
- `--qsa-local <int>` — Always-kept recent local KV tokens (default: `256`)
- `--qsa-sink <int>` — Always-kept initial attention-sink tokens (default: `16`)
- `STRATA_NO_TP_GRAPH=1` — Disable Dual `hipGraph` (run Push-only P2P on direct streams)
- `STRATA_NO_P2P_SPIN=1` — Fall back from FineGrained P2P spin-barrier to `hipEvent` + `hipMemcpyPeerAsync`
- `STRATA_NO_TP_MOE=1` — Fall back from expert-internal TP=2 (`576+576`) to 64+64 EP split
- `STRATA_NO_CTRL_F16=1` — Keep loop controllers in F32 instead of FP16
- `STRATA_EV_PROF=1` — Print zero-host-sync per-stage GPU timing breakdown (with `STRATA_NO_TP_GRAPH=1`)

## License

MIT — see [LICENSE](LICENSE). Upstream credits: [Niko1221/Strata](https://github.com/Niko1221/Strata) and [kuraneko1/Strata](https://github.com/kuraneko1/Strata).
