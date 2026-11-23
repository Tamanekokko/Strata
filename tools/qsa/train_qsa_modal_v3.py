import os
import struct
import time
import math
import modal

app = modal.App("inkling2-qsa-distill-v3")
vol = modal.Volume.from_name("inkling2-qsa-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "numpy")
)


def read_qsat_meta(path: str):
    with open(path, "rb") as f:
        hdr = struct.unpack("<6i", f.read(24))
        magic, version, n_full, n_embd, max_blocks, _ = hdr
        assert magic == 0x51534154, f"Bad magic {hex(magic)}"
        full_vls = list(struct.unpack(f"<{n_full}i", f.read(4 * n_full)))
    return full_vls, n_embd, max_blocks


def read_qsat_layer(path: str, target_fidx: int):
    import numpy as np
    with open(path, "rb") as f:
        hdr = struct.unpack("<6i", f.read(24))
        magic, version, n_full, n_embd, max_blocks, _ = hdr
        f.seek(4 * n_full, os.SEEK_CUR)
        for fi in range(n_full):
            nb, nq = struct.unpack("<2i", f.read(8))
            sz_k = nb * n_embd * 2
            sz_qp = nq * 4
            sz_qf = nq * n_embd * 2
            sz_pt = nq * max_blocks * 2
            if fi == target_fidx:
                k_mean = np.frombuffer(f.read(sz_k), dtype=np.float16).reshape(nb, n_embd).copy()
                q_pos = np.frombuffer(f.read(sz_qp), dtype=np.int32).copy()
                q_feat = np.frombuffer(f.read(sz_qf), dtype=np.float16).reshape(nq, n_embd).copy()
                p_tgt = np.frombuffer(f.read(sz_pt), dtype=np.float16).reshape(nq, max_blocks).copy()
                return k_mean, q_pos, q_feat, p_tgt
            else:
                f.seek(sz_k + sz_qp + sz_qf + sz_pt, os.SEEK_CUR)
    raise ValueError(f"Layer {target_fidx} not found in {path}")


@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    volumes={"/data": vol},
    timeout=3600,
)
def train_all_layers_v3(
    epochs: int = 100,
    lr: float = 3e-3,
    h_idx: int = 4,
    d_idx: int = 128,
    rope_theta: float = 0.0,
    sink_blocks: int = 4,
    local_blocks: int = 64,
    sharp_weight: float = 0.30,
    tail_weight: float = 2.0,
    teacher_dir: str = "/data/teacher_v3",
    out_name: str = "inkling2-qsa-indexer-v3.bin",
    n_val: int = 8,
):
    import glob
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device("cuda")
    files = sorted(glob.glob(f"{teacher_dir}/seq_*.qsat"))
    assert len(files) > n_val, f"Need > {n_val} qsat files in {teacher_dir}, got {len(files)}"
    n_train = len(files) - n_val
    full_vls, n_embd, max_blocks = read_qsat_meta(files[0])
    n_full = len(full_vls)
    print(
        f"v3 Distill: {len(files)} seqs (train={n_train}, val={n_val}), "
        f"n_full={n_full} vls={full_vls} theta={rope_theta} sink={sink_blocks} loc={local_blocks}",
        flush=True,
    )

    half_d = d_idx // 2
    inv_freq = (
        (1.0 / (rope_theta ** (torch.arange(0, half_d, device=device, dtype=torch.float32) / half_d)))
        if rope_theta > 0.0
        else None
    )

    def apply_rope(x, pos):
        if rope_theta <= 0.0:
            return x
        angles = pos.float().unsqueeze(-1) * inv_freq
        c = torch.cos(angles)
        s = torch.sin(angles)
        while c.ndim < x.ndim:
            c = c.unsqueeze(-2)
            s = s.unsqueeze(-2)
        x0 = x[..., :half_d]
        x1 = x[..., half_d:]
        return torch.cat([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1)

    class QsaIndexerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(n_embd, h_idx * d_idx, bias=False)
            self.wk = nn.Linear(n_embd, d_idx, bias=False)
            # Positive-aligned init so q·k starts positive and never triggers dying ReLU
            with torch.no_grad():
                wk_w = self.wk.weight.data
                std = wk_w.std().clamp_min(1e-4)
                for h in range(h_idx):
                    self.wq.weight.data[h * d_idx : (h + 1) * d_idx].copy_(
                        wk_w + 0.20 * std * torch.randn_like(wk_w)
                    )
            self.q_norm_w = nn.Parameter(torch.ones(d_idx))
            self.k_norm_w = nn.Parameter(torch.ones(d_idx))
            self.w_head_raw = nn.Parameter(torch.full((h_idx,), 1.0))

        def rms_norm(self, x, w):
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
            return (x.float() * torch.rsqrt(var + 1e-6)) * w

        def forward(self, q_feat, q_pos, k_mean, leaky_slope: float = 0.0):
            nq = q_feat.shape[0]
            nb = k_mean.shape[0]
            q = self.wq(q_feat).view(nq, h_idx, d_idx)
            k = self.wk(k_mean)
            q = self.rms_norm(q, self.q_norm_w)
            k = self.rms_norm(k, self.k_norm_w)
            q = apply_rope(q, q_pos)
            b_pos = torch.arange(nb, device=q.device, dtype=torch.int32) * 4
            k = apply_rope(k, b_pos)
            dots = torch.einsum("qhd,bd->qhb", q, k) * (1.0 / math.sqrt(d_idx))
            act = F.leaky_relu(dots, negative_slope=leaky_slope) if leaky_slope > 0.0 else F.relu(dots)
            wh = F.softplus(self.w_head_raw).view(1, h_idx, 1)
            return (act * wh).sum(dim=1)

    trained_weights = []
    summary_lines = []

    for f_idx in range(n_full):
        t0 = time.time()
        train_seqs = []
        val_seqs = []
        for si, fp in enumerate(files):
            k_mean, q_pos, q_feat, p_tgt = read_qsat_layer(fp, f_idx)
            nb = k_mean.shape[0]
            kt = torch.from_numpy(k_mean).to(device=device, dtype=torch.bfloat16)
            qf = torch.from_numpy(q_feat).to(device=device, dtype=torch.bfloat16)
            qp = torch.from_numpy(q_pos).to(device=device)
            pt = torch.from_numpy(p_tgt[:, :nb]).to(device=device, dtype=torch.float32)

            n_valid_b = (qp + 1) // 4
            b_idx = torch.arange(nb, device=device).unsqueeze(0)
            causal_ok = b_idx < n_valid_b.unsqueeze(1)
            sink_loc = (b_idx < sink_blocks) | ((b_idx >= n_valid_b.unsqueeze(1) - local_blocks) & causal_ok)
            remote_ok = causal_ok & (~sink_loc)

            pt_c = pt * causal_ok.float()
            pt_c = pt_c / (pt_c.sum(dim=-1, keepdim=True) + 1e-8)

            pt_rem = pt_c * remote_ok.float()
            rem_mass = pt_rem.sum(dim=-1)
            valid_q = (rem_mass > 0.01) & (remote_ok.sum(dim=-1) > 16)

            if valid_q.sum().item() == 0:
                continue

            qf = qf[valid_q]
            qp = qp[valid_q]
            remote_ok = remote_ok[valid_q]
            pt_rem_norm = pt_rem[valid_q] / (rem_mass[valid_q].unsqueeze(-1) + 1e-8)

            # Sharp target distribution to emphasize needle / code-identifier spikes
            pt_sharp = pt_rem_norm.clamp_min(0.0).pow(1.5)
            pt_sharp = pt_sharp / (pt_sharp.sum(dim=-1, keepdim=True) + 1e-8)

            # Higher weight for tail QA queries (last 256 tokens of sequence)
            seq_end = nb * 4
            qw = torch.where(qp >= (seq_end - 256), torch.full_like(qp, tail_weight, dtype=torch.float32), torch.ones_like(qp, dtype=torch.float32))
            qw = qw / qw.mean()

            item = (kt, qp, qf, pt_rem_norm, pt_sharp, remote_ok, qw)
            if si < n_train:
                train_seqs.append(item)
            else:
                val_seqs.append(item)

        best_metric = -1.0
        best_r128 = 0.0
        best_r256 = 0.0
        best_r444 = 0.0
        best_state = None

        for attempt in range(3):
            torch.manual_seed(42 + f_idx * 100 + attempt * 7)
            model = QsaIndexerLayer().to(device=device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr * (0.5 ** attempt), weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)
            dead_retry = False

            for ep in range(epochs):
                model.train()
                leaky = 0.05 * max(0.0, 1.0 - ep / max(1, int(epochs * 0.10)))
                perm = torch.randperm(len(train_seqs)).tolist()
                tr_loss = 0.0
                for si in perm:
                    kt, qp, qf, pt_rem, pt_sharp, remote_ok, qw = train_seqs[si]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        scores = model(qf, qp, kt, leaky_slope=leaky).float()
                    scores = scores.masked_fill(~remote_ok, -1e9)
                    log_p = F.log_softmax(scores, dim=-1)
                    kl_base = (pt_rem * (torch.log(pt_rem.clamp_min(1e-8)) - log_p)).sum(dim=-1)
                    kl_sharp = (pt_sharp * (torch.log(pt_sharp.clamp_min(1e-8)) - log_p)).sum(dim=-1)
                    loss = (((1.0 - sharp_weight) * kl_base + sharp_weight * kl_sharp) * qw).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    tr_loss += loss.item()
                sched.step()

                if (ep + 1) % 20 == 0 or ep == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        val_kl = 0.0
                        r128_l, r256_l, r444_l = [], [], []
                        for kt, qp, qf, pt_rem, pt_sharp, remote_ok, qw in val_seqs:
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                scores = model(qf, qp, kt, leaky_slope=0.0).float()
                            scores = scores.masked_fill(~remote_ok, -1e9)
                            log_p = F.log_softmax(scores, dim=-1)
                            val_kl += (pt_rem * (torch.log(pt_rem.clamp_min(1e-8)) - log_p)).sum(dim=-1).mean().item()
                            nb = scores.shape[1]
                            t128 = torch.topk(scores, k=min(128, nb), dim=-1).indices
                            t256 = torch.topk(scores, k=min(256, nb), dim=-1).indices
                            t444 = torch.topk(scores, k=min(444, nb), dim=-1).indices
                            r128_l.append(pt_rem.gather(1, t128).sum(dim=-1).mean().item())
                            r256_l.append(pt_rem.gather(1, t256).sum(dim=-1).mean().item())
                            r444_l.append(pt_rem.gather(1, t444).sum(dim=-1).mean().item())
                        r128 = 100.0 * sum(r128_l) / len(r128_l)
                        r256 = 100.0 * sum(r256_l) / len(r256_l)
                        r444 = 100.0 * sum(r444_l) / len(r444_l)
                        metric = 0.5 * r128 + 0.5 * r256
                        if metric > best_metric:
                            best_metric = metric
                            best_r128, best_r256, best_r444 = r128, r256, r444
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        if ep == epochs - 1 or (ep + 1) in (20, 40, 60):
                            print(
                                f"  vl={full_vls[f_idx]:2d} att={attempt} ep={ep+1:3d} tr={tr_loss/len(train_seqs):.4f} "
                                f"val_kl={val_kl/len(val_seqs):.4f} rem@128={r128:.2f}% rem@256={r256:.2f}% rem@444={r444:.2f}%",
                                flush=True,
                            )
                        if ep == 39 and ((tr_loss / len(train_seqs)) > 1.2 or r128 < 20.0):
                            print(f"  vl={full_vls[f_idx]:2d} att={attempt} dead/stuck (tr={tr_loss/len(train_seqs):.3f}, r128={r128:.1f}%), retrying...", flush=True)
                            dead_retry = True
                            break
            if not dead_retry:
                break

        dt = time.time() - t0
        wh_final = F.softplus(best_state["w_head_raw"]).numpy().astype(np.float32)
        wq_fp16 = best_state["wq.weight"].to(torch.float16).numpy()
        wk_fp16 = best_state["wk.weight"].to(torch.float16).numpy()
        qn_f32 = best_state["q_norm_w"].float().numpy()
        kn_f32 = best_state["k_norm_w"].float().numpy()
        trained_weights.append((wq_fp16, wk_fp16, qn_f32, kn_f32, wh_final))
        line = (
            f"vl={full_vls[f_idx]:2d}: best rem@128={best_r128:.2f}% rem@256={best_r256:.2f}% "
            f"rem@444={best_r444:.2f}% w_head={np.round(wh_final, 3).tolist()} ({dt:.1f}s)"
        )
        summary_lines.append(line)
        print(f"DONE {line}", flush=True)

        del train_seqs, val_seqs
        torch.cuda.empty_cache()

    out_path = f"/data/{out_name}"
    with open(out_path, "wb") as f:
        f.write(struct.pack("<7if", 0x51534157, 1, n_full, n_embd, h_idx, d_idx, 4, float(rope_theta)))
        f.write(struct.pack(f"<{n_full}i", *full_vls))
        for wq, wk, qn, kn, wh in trained_weights:
            f.write(wq.tobytes())
            f.write(wk.tobytes())
            f.write(qn.tobytes())
            f.write(kn.tobytes())
            f.write(wh.tobytes())
    vol.commit()
    sz = os.path.getsize(out_path)
    print(f"Saved {out_path} ({sz / (1024*1024):.2f} MiB)", flush=True)
    return "\n".join(summary_lines)


@app.local_entrypoint()
def main(
    epochs: int = 100,
    lr: float = 3e-3,
    rope_theta: float = 0.0,
    out_name: str = "inkling2-qsa-indexer-v3.bin",
    teacher_dir: str = "/data/teacher_v3",
):
    res = train_all_layers_v3.remote(
        epochs=epochs, lr=lr, rope_theta=rope_theta, out_name=out_name, teacher_dir=teacher_dir
    )
    print("\n=== V3 TRAINING SUMMARY ===")
    print(res)
