#!/usr/bin/env python3
import torch, os
import numpy as np

###############################
# 1) Paths (edit if needed)  #
###############################
scgs_pth         = "data/load_from_ply/chkpnt20000.pth"
reference_3dgrut = "runs/hilbert_test/ckpt_last.pt"   # A valid 3DGRUT checkpoint
out_3dgrut_pt    = "data/load_from_ply/chkpnt20000_3dgrut.pt"

print("Working directory:", os.getcwd())
print("Loading SCGS checkpoint from:", scgs_pth)
scgs_ckpt = torch.load(scgs_pth, map_location="cpu", weights_only=False)

# SCGS.save does: torch.save((gaussians.capture(), iteration), …)
# so scgs_ckpt is (capture_tuple, iteration)
capture_tuple, scgs_iter = scgs_ckpt
print(" SCGS capture tuple length:", len(capture_tuple))
print(" SCGS iteration (we will ignore for 3DGRUT):", scgs_iter)

# Inspect SCGS capture tuple
for i, elem in enumerate(capture_tuple):
    tname = type(elem).__name__
    shape = tuple(elem.shape) if isinstance(elem, torch.Tensor) else None
    print(f"  [SCGS idx {i:2d}] → {tname:>20s}  shape={shape}")

# According to SCGS capture():
# idx0 = active_sh_degree (int or small tensor)   → ignore
# idx1 = _xyz                (Tensor [N,3])      → “positions”
# idx2 = _features_dc        (Tensor [N,1,3])    → needs to become (N,3) for “features_albedo”
# idx3 = _features_rest      (Tensor [N,15,3])   → reduce to (N,3) for “features_specular”
# idx4 = _scaling            (Tensor [N,3])      → “scale”
# idx5 = _rotation           (Tensor [N,4])      → quaternion; need axis‐angle (N,3) for “rotation”
# idx6 = _opacity            (Tensor [N,1])      → “density”
# idx7 = max_radii2D         (Tensor [N,])       → ignore
# idx8 = xyz_gradient_accum  (Tensor [N,1])      → ignore
# idx9 = denom               (Tensor [N,1])      → ignore
# idx10 = optimizer.state_dict() (dict)          → “optimizer_state_dict”
# idx11 = spatial_lr_scale   (float or Tensor)   → ignore

# 2) Extract SCGS tensors:
xyz               = capture_tuple[1]   # shape [N,3]
features_dc       = capture_tuple[2]   # shape [N,1,3]
features_rest     = capture_tuple[3]   # shape [N,15,3]
scaling           = capture_tuple[4]   # shape [N,3]
rotation_quat     = capture_tuple[5]   # shape [N,4]
opacity           = capture_tuple[6]   # shape [N,1]
optimizer_state   = capture_tuple[10]  # dict

# Convert everything to non‐grad (if needed)
xyz           = xyz.detach()
features_dc   = features_dc.detach()
features_rest = features_rest.detach()
scaling       = scaling.detach()
rotation_quat = rotation_quat.detach()
opacity       = opacity.detach()

# 3) Number of Gaussians, etc.
N = xyz.shape[0]
n_active_features = torch.tensor(N, dtype=torch.int64)
max_n_features    = torch.tensor(N, dtype=torch.int64)

# 4) Compute “scene_extent” exactly as 3DGRUT expects: (xmin, xmax, ymin, ymax, zmin, zmax)
xyz_np = xyz.cpu().numpy()
mins   = xyz_np.min(axis=0)  # [min_x, min_y, min_z]
maxs   = xyz_np.max(axis=0)  # [max_x, max_y, max_z]
scene_extent = torch.tensor([mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]],
                            dtype=torch.float32)

# 5) Convert SCGS features → shapes that 3DGRUT needs

# 5a) features_albedo: SCGS’s features_dc is (N,1,3). 3DGRUT wants (N,3). So:
features_albedo = features_dc.squeeze(1)   # now (N,3)

# 5b) features_specular: SCGS’s features_rest is (N,15,3). 3DGRUT wants (N,3).
#    We’ll simply take the DC component (index 0) of the SH coefficients:
features_specular = features_rest[:, 0, :]    # now (N,3)

# 6) Convert SCGS quaternion (N,4) → axis‐angle (N,3), since 3DGRUT expects rotation shape (N,3)
#    Quaternion format is assumed [x, y, z, w] (check SCGS’s convention; adapt if reversed)
q = rotation_quat.cpu().numpy()   # shape [N,4]
# Ensure shape (N,4)
assert q.shape[1] == 4, "Expected SCGS rotation to be shape [N,4]"

# Function to convert a unit quaternion [x,y,z,w] → axis‐angle (3)
def quaternion_to_axisangle(quat: np.ndarray):
    """
    Input: quat is shape (4,) with [x, y, z, w].
    Output: axis‐angle vector of shape (3,) = axis * angle, where angle ∈ [0, π].
    """
    x, y, z, w = quat
    # Normalize (just in case)
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    x /= norm; y /= norm; z /= norm; w /= norm

    # Angle:
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(1.0 - w*w)
    if s < 1e-8:
        # If s is small, direction can be anything (use x,y,z directly)
        return np.array([x*angle, y*angle, z*angle], dtype=np.float32)
    else:
        # Axis = [x/s, y/s, z/s]
        axis = np.array([x/s, y/s, z/s], dtype=np.float32)
        return axis * angle

# Apply conversion to all N quaternions
axisangle = np.zeros((N, 3), dtype=np.float32)
for i in range(N):
    axisangle[i] = quaternion_to_axisangle(q[i])
rotation = torch.from_numpy(axisangle)  # shape [N,3]

# 7) density = SCGS opacity (N,1)
density = opacity  # already shape [N,1]

# 8) Now load reference 3DGRUT checkpoint to grab all “extra” keys
print("\nLoading reference 3DGRUT checkpoint from:", reference_3dgrut)
ref = torch.load(reference_3dgrut, map_location="cpu")

print("Keys in reference 3DGRUT checkpoint:")
for k in sorted(ref.keys()):
    print("  ", k)

# 9) Build the new 3DGRUT checkpoint dict
new_ckpt = {
    "positions":                 xyz,                   # (N,3)
    "rotation":                  rotation,              # (N,3)
    "scale":                     scaling,               # (N,3)
    "density":                   density,               # (N,1)
    "features_albedo":           features_albedo,       # (N,3)
    "features_specular":         features_specular,     # (N,3)
    "n_active_features":         n_active_features,     # int or Tensor
    "max_n_features":            max_n_features,        # int or Tensor
    "scene_extent":              scene_extent,          # (6,)
    # Copy SCGS’s optimizer if you want to resume exactly
    "optimizer_state_dict":      optimizer_state,       # SCGS’s optimizer → 3DGRUT reuses it
}

# If 3DGRUT expects key “optimizer” instead of “optimizer_state_dict”, do:
if "optimizer" in ref or "optimizer_state" in ref:
    # Some 3DGRUT versions call it “optimizer_state” or “optimizer”
    new_ckpt["optimizer"] = optimizer_state

# Copy all extra keys from the reference that init_from_checkpoint expects:
extra_keys = [
    "feature_dim_increase_interval",
    "feature_dim_increase_step",
    "background",
    "learning_rate",
    "scale_schedule",
    "iteration",
    # ...add any other keys you saw in the inspect script, e.g. “sampler_state_dict”
]

for key in extra_keys:
    if key in ref:
        new_ckpt[key] = ref[key]
    else:
        print(f"⚠️ Warning: reference does NOT contain key '{key}'")

# 10) Final check: print out new_ckpt keys & expected shapes
print("\nFinal keys in new 3DGRUT checkpoint:")
for k in sorted(new_ckpt.keys()):
    v = new_ckpt[k]
    if isinstance(v, torch.Tensor):
        print(f"  {k:30s} → Tensor   shape: {tuple(v.shape)}")
    else:
        print(f"  {k:30s} → {type(v).__name__}")

# 11) Save the merged checkpoint
print("\nSaving merged 3DGRUT checkpoint to:", out_3dgrut_pt)
torch.save(new_ckpt, out_3dgrut_pt)
print("✅ Done.")
