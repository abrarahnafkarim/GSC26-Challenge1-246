"""Recover the organizers' real trigger from the backdoored clients.

defense/visible_case contains 10 clients of one cohort. Two of them (3 and 5)
collude along a shared direction and are backdoored with the REAL trigger; the
other eight are honest. That contrast is exactly what Neural-Cleanse-style
trigger recovery needs, and unlike our surrogate pipeline it requires no data:
we optimize a mask + pattern that drives the BACKDOORED clients toward a class
while leaving the HONEST clients unmoved. Only a feature the backdoored models
learned and the honest ones did not can do that.

Because SmallCNN ends in AdaptiveAvgPool2d((1,1)) the input size is free, so the
recovered mask's position and extent are readable directly: sunglasses should
concentrate over the eye band, a surgical mask over the lower face.

This tells us the trigger's position, size and colour - the three things our
hand-drawn ellipses most likely got wrong, and the reason the surrogate-trained
backdoor lowered ASR instead of raising it.

Usage:
    python solution/recover_trigger.py                # all 4 classes
    python solution/recover_trigger.py --target 0 --steps 400
"""

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import SmallCNN  # noqa: E402

BACKDOORED = (3, 5)


def load_cohort(device, n_honest=None):
    files = sorted((ROOT / "defense" / "visible_case").glob("client_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    nets = []
    for path in files:
        net = SmallCNN()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.to(device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        nets.append(net)
    bad = [nets[i] for i in BACKDOORED]
    good = [n for i, n in enumerate(nets) if i not in BACKDOORED]
    if n_honest is not None:
        good = good[:n_honest]   # a few honest nets suffice as the "unmoved" set
    return bad, good


def recover(target, size, steps, lam, batch, lr, device, seed=0, n_honest=None,
           tv=0.01):
    """Neural-Cleanse-style trigger reversal against the backdoored models.

    Only bad (backdoored) models drive the objective: minimize CE toward
    `target` under a strong L1 mask penalty, exactly as in the original Neural
    Cleanse recipe (data-free / random-input variant). An earlier version also
    subtracted the honest models' score from the objective; that let the
    optimizer satisfy the loss by crashing BOTH sides toward zero via an
    unbounded adversarial blackout (mask area ~0.98, extreme pixel values)
    instead of finding the real, small, colored trigger. The honest models are
    used only afterward, to check the recovered patch does NOT fool them.
    """
    torch.manual_seed(seed)
    bad, good = load_cohort(device, n_honest=n_honest)

    # Start the mask near zero (logit -4 -> ~0.018) so it can only GROW where it
    # genuinely helps.
    mask_logit = torch.full((1, 1, size, size), -4.0,
                            device=device).requires_grad_(True)
    pattern_raw = torch.zeros(1, 3, size, size, device=device).requires_grad_(True)
    opt = torch.optim.Adam([mask_logit, pattern_raw], lr=lr)
    target_lbl = torch.full((batch,), target, dtype=torch.long, device=device)

    for step in range(steps):
        # Random face-free backgrounds: the trigger must work regardless of the
        # image it is pasted on, which is what makes it a backdoor.
        x = torch.rand(batch, 3, size, size, device=device)   # bounded [0,1]
        mask = torch.sigmoid(mask_logit)
        pattern = torch.sigmoid(pattern_raw)                   # bounded [0,1]
        x_trig = (1 - mask) * x + mask * pattern

        ce = sum(torch.nn.functional.cross_entropy(n(x_trig), target_lbl)
                 for n in bad) / len(bad)
        # Total variation: a real physical trigger (sunglasses, mask) is a
        # contiguous blob, not scattered pixels.
        tv_loss = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean() + \
                  (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean()
        loss = ce + lam * mask.mean() + tv * tv_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        mask = torch.sigmoid(mask_logit)
        pattern = torch.sigmoid(pattern_raw)
        x = torch.rand(256, 3, size, size, device=device)
        x_trig = (1 - mask) * x + mask * pattern
        p_bad = torch.stack([n(x_trig).softmax(1)[:, target] for n in bad]).mean()
        p_good = torch.stack([n(x_trig).softmax(1)[:, target] for n in good]).mean()
        base_bad = torch.stack([n(x).softmax(1)[:, target] for n in bad]).mean()
        base_good = torch.stack([n(x).softmax(1)[:, target] for n in good]).mean()
    return {
        "mask": mask.detach()[0, 0].cpu(),
        "pattern": pattern.detach()[0].cpu(),
        "p_bad": float(p_bad), "p_good": float(p_good),
        "base_bad": float(base_bad), "base_good": float(base_good),
        "l1": float(mask.mean()),
    }


def heatmap(mask, rows=16):
    """Coarse ASCII view of where the mask concentrates."""
    size = mask.shape[0]
    step = max(1, size // rows)
    pooled = mask.reshape(size // step, step, size // step, step).mean(dim=(1, 3))
    pooled = pooled / (pooled.max() + 1e-12)
    ramp = " .:-=+*#%@"
    lines = []
    for r in range(pooled.shape[0]):
        lines.append("".join(ramp[min(len(ramp) - 1, int(v * len(ramp)))]
                             for v in pooled[r]))
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=None,
                   help="class to recover for (default: try all 4)")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lam", type=float, default=0.03, help="mask L1 penalty")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--honest", type=int, default=4,
                   help="how many honest nets to use as post-hoc validation")
    p.add_argument("--tv", type=float, default=0.01,
                   help="total-variation penalty (favors a contiguous blob)")
    p.add_argument("--out", type=Path,
                   default=ROOT / "solution" / "directions" / "trigger_recovery.txt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    targets = [args.target] if args.target is not None else [0, 1, 2, 3]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.open("w")

    def emit(text=""):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    emit(f"trigger recovery | size={args.size} steps={args.steps} "
         f"honest={args.honest} device={device}")
    results = {}
    for t in targets:
        r = recover(t, args.size, args.steps, args.lam, args.batch, args.lr,
                    device, n_honest=args.honest, tv=args.tv)
        results[t] = r
        torch.save({"mask": r["mask"], "pattern": r["pattern"], "target": t},
                   args.out.parent / f"trigger_class{t}.pt")
        emit(f"\n=== target class {t} ===")
        emit(f"  p(target) backdoored {r['base_bad']:.3f} -> {r['p_bad']:.3f}"
             f"   honest {r['base_good']:.3f} -> {r['p_good']:.3f}")
        emit(f"  separation = {r['p_bad'] - r['p_good']:+.3f}"
             f"   mask area = {r['l1']:.3f}")
        for line in heatmap(r["mask"]):
            emit("   |" + line + "|")

    if len(results) > 1:
        emit("\n=== summary (largest separation = the implanted class) ===")
        for t, r in sorted(results.items(),
                           key=lambda kv: -(kv[1]["p_bad"] - kv[1]["p_good"])):
            emit(f"  class {t}: separation {r['p_bad'] - r['p_good']:+.3f}"
                 f"  mask area {r['l1']:.3f}")
    log.close()


if __name__ == "__main__":
    main()
