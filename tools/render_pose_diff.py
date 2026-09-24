"""Same-pose diff: playground engine render vs trainer render vs GT.

Renders a chosen val frame of the FIORD meetingroom through the PLAYGROUND
model path (hydra 3dgrt conf -> MixtureOfGaussians.init_from_ply ->
build_acc -> model.trace, mirroring engine.py:1119-1145 load and the
no-primitives branch of render_pass, engine.py:996-997) with the EXACT
OPENCV_FISHEYE rays + pose the trainer eval used (same ColmapDataset item),
plus a 90-deg pinhole from the same seat, plus a monster-pruned variant,
plus a trainer-conf 3DGUT reproduction as harness check.

Run (GPU, ~1 min):
  python tools/render_pose_diff.py --run_dir runs/<experiment>/<run> [--dataset <path>]
--dataset defaults to $VILOTA_DATASETS/meetingroom. Optional: --frame <val idx> --prune_smax 5.0

Outputs in <run_dir>/pose_diff/: playground_<frame>.png,
playground_pinhole_<frame>.png, pruned_<frame>.png, pruned_pinhole_<frame>.png,
trainer_repro_<frame>.png, gt_<frame>.png + PSNR table on stdout.

CPU-verified prediction (fog census, Sep 14 2026): the model holds 1379
splats with s_max>5 & opa>0.3, all centered OUTSIDE the room; volumetric
alpha at every camera seat totals ~1.0 under the deg-4 3DGRT kernel
(0.94-0.97 deg-2). 3DGUT sorts them at center depth (~500 units, behind
walls) so trainer eval never sees them; the ray tracer composites them at
the ray's closest approach -> fog. Expect: playground_* soup, pruned_*
coherent, trainer_repro ~= runs render.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def masked_psnr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    # a, b: (H,W,3) in [0,1]; mask: (H,W,1) 0/1
    mse = (((a - b) * mask) ** 2).sum() / (mask.sum() * 3)
    return float(-10.0 * torch.log10(mse))


def save_png(path: Path, img: torch.Tensor) -> None:
    from PIL import Image
    arr = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(arr).save(path)


def load_playground_model(ply_path: str, config_name: str):
    """Mirrors Engine3DGRUT.load_3dgrt_object's .ply branch (engine.py:1119-1145)."""
    from hydra import compose, initialize_config_dir
    from threedgrut.model.model import MixtureOfGaussians
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        conf = compose(config_name=config_name)
    model = MixtureOfGaussians(conf)
    model.init_from_ply(ply_path, init_model=False)
    model.build_acc(rebuild=True)
    return model


def write_pruned_ply(src: str, dst: str, smax_thr: float, opa_thr: float) -> int:
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(src)
    v = ply.elements[0]
    smax = np.exp(np.stack([np.asarray(v[f"scale_{i}"]) for i in range(3)], 1)).max(1)
    opa = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"])))
    keep = ~((smax > smax_thr) & (opa > opa_thr))
    PlyData([PlyElement.describe(v.data[keep], "vertex")]).write(dst)
    return int((~keep).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="training run folder, runs/<experiment>/<run>")
    datasets = os.environ.get("VILOTA_DATASETS")
    ap.add_argument("--dataset", default=datasets and os.path.join(datasets, "meetingroom"),
                    required=datasets is None,
                    help="FIORD meetingroom dataset (default $VILOTA_DATASETS/meetingroom)")
    ap.add_argument("--frame", type=int, default=4, help="val-split index; renders/<frame:05d>.png")
    ap.add_argument("--config", default="apps/colmap_3dgrt.yaml", help="playground default_gs_config")
    ap.add_argument("--prune_smax", type=float, default=5.0)
    ap.add_argument("--prune_opa", type=float, default=0.3)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ply_path = str(run_dir / "export_last.ply")
    out_dir = run_dir / "pose_diff"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.frame:05d}"

    # --- exact eval rays / pose / GT / mask (identical to trainer eval batch) ---
    from threedgrut.datasets.dataset_colmap import ColmapDataset
    ds = ColmapDataset(args.dataset, device="cuda", split="val", downsample_factor=4)
    print(f"frame {tag} = {os.path.basename(ds.image_paths[args.frame])}")
    item = ds[args.frame]
    batch = torch.utils.data.default_collate([item])
    gpu_batch = ds.get_gpu_batch_with_intrinsics(batch)
    H, W = gpu_batch.rgb_gt.shape[1:3]
    gt = gpu_batch.rgb_gt[0]                       # (H,W,3)
    mask = gpu_batch.mask[0] if gpu_batch.mask is not None else torch.ones(H, W, 1, device="cuda")
    pose = gpu_batch.T_to_world                    # (1,4,4) C2W
    print(f"seat C = {pose[0, :3, 3].tolist()}")
    save_png(out_dir / f"gt_{tag}.png", gt)

    # 90-deg pinhole rays from the same seat (f = W/2), COLMAP camera convention
    from threedgrut.datasets.utils import pinhole_camera_rays
    u = np.tile(np.arange(W), H)
    v = np.arange(H).repeat(W)
    po, pd = pinhole_camera_rays(u, v, W / 2, W / 2, W, H, None)
    pin_o = torch.tensor(po, dtype=torch.float32, device="cuda").reshape(1, H, W, 3)
    pin_d = torch.tensor(pd, dtype=torch.float32, device="cuda").reshape(1, H, W, 3)

    trainer_png = run_dir / "ours_30000" / "renders" / f"{tag}.png"
    trainer_img = None
    if trainer_png.exists():
        from PIL import Image
        trainer_img = torch.tensor(
            np.asarray(Image.open(trainer_png))[..., :3] / 255.0, dtype=torch.float32, device="cuda"
        )

    results = {}

    def trace_and_save(model, name, rays_o, rays_d):
        with torch.no_grad():
            rb = model.trace(rays_o=rays_o, rays_d=rays_d, T_to_world=pose)
        img = rb["pred_rgb"][0].clamp(0, 1)
        save_png(out_dir / f"{name}_{tag}.png", img)
        results[name] = img
        return img

    # --- A/B: playground path, default 3dgrt conf (deg-4 kernel) ---
    pg = load_playground_model(ply_path, args.config)
    trace_and_save(pg, "playground", gpu_batch.rays_ori, gpu_batch.rays_dir)
    trace_and_save(pg, "playground_pinhole", pin_o, pin_d)
    del pg
    torch.cuda.empty_cache()

    # --- C: monsters pruned, same playground path ---
    pruned_ply = str(out_dir / "export_last_pruned.ply")
    n_pruned = write_pruned_ply(ply_path, pruned_ply, args.prune_smax, args.prune_opa)
    print(f"pruned {n_pruned} splats with s_max>{args.prune_smax} & opa>{args.prune_opa}")
    pr = load_playground_model(pruned_ply, args.config)
    trace_and_save(pr, "pruned", gpu_batch.rays_ori, gpu_batch.rays_dir)
    trace_and_save(pr, "pruned_pinhole", pin_o, pin_d)
    del pr
    torch.cuda.empty_cache()

    # --- D: trainer-conf 3DGUT reproduction (harness check vs renders/<tag>.png) ---
    from omegaconf import OmegaConf
    from threedgrut.model.model import MixtureOfGaussians
    conf_t = OmegaConf.load(run_dir / "parsed.yaml")
    mt = MixtureOfGaussians(conf_t)
    mt.init_from_ply(ply_path, init_model=False)
    mt.build_acc(rebuild=True)
    with torch.no_grad():
        out = mt(gpu_batch, train=False)
    img = out["pred_rgb"][0].clamp(0, 1)
    save_png(out_dir / f"trainer_repro_{tag}.png", img)
    results["trainer_repro"] = img

    # --- metrics ---
    print(f"\nmasked PSNR (dB), frame {tag}, mask valid frac {mask.mean():.3f}:")
    for name in ("playground", "pruned", "trainer_repro"):
        line = f"  {name:14s} vs GT: {masked_psnr(results[name], gt, mask):6.2f}"
        if trainer_img is not None:
            line += f"   vs trainer render: {masked_psnr(results[name], trainer_img, mask):6.2f}"
        print(line)
    if trainer_img is not None:
        print(f"  trainer render vs GT (sanity): {masked_psnr(trainer_img, gt, mask):6.2f}")
    print("\ninterpretation: see CLAUDE.md 'FIORD playground contradiction'")


if __name__ == "__main__":
    main()
