"""Standalone verification of estimate_theta_star (fisheye.py).

Forward model (Kannala-Brandt 4, as used by vk_calibrate / basalt):
    ru(theta) = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9

Test: take CamA's k from calibration_files/vk180.json, build ru from a sweep
of theta, feed ru back through estimate_theta_star, compare recovered theta
to the input theta.  Also probe (a) the units of the LUT search range and
(b) behaviour past the lens limit.

Run:  python tests/test_estimate_theta_star.py
Fails (AssertionError) if the round trip is worse than 1e-5 rad or 0.01 px,
if the lookup still inverts the pre-fix theta^2 polynomial, or if radii past
the lookup table extrapolate beyond pi.
"""

import json
import math
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from threedgrut_playground.utils.kaolin_future.fisheye import (  # noqa: E402
    estimate_theta_star,
)


def load_cam(idx):
    with open(os.path.join(REPO, "calibration_files", "vk180.json")) as f:
        calib = json.load(f)
    for cam_id, cam in calib["cameraData"]:
        if cam_id == idx:
            k = cam["distortionCoeff"][:4]
            K = cam["intrinsicMatrix"]
            return k, (K[0][0], K[1][1], K[0][2], K[1][2]), cam["width"], cam["height"]
    raise KeyError(idx)


def forward_ru(theta, k):
    """The TRUE KB4 forward map.  Odd powers 3,5,7,9."""
    k1, k2, k3, k4 = k
    return theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9


def forward_ru_buggy(theta, k):
    """The pre-fix polynomial: `k1 * theta**2` instead of `theta**3` (bug 4)."""
    k1, k2, k3, k4 = k
    return theta + k1 * theta**2 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9


def dru_dtheta(theta, k):
    k1, k2, k3, k4 = k
    return 1 + 3 * k1 * theta**2 + 5 * k2 * theta**4 + 7 * k3 * theta**6 + 9 * k4 * theta**8


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    torch.set_printoptions(precision=6)
    k, (fx, fy, cx, cy), W, H = load_cam(0)
    k1, k2, k3, k4 = k
    print(f"CamA (cameraData id 0): {W}x{H}  fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"k = [{k1:.6f}, {k2:.6f}, {k3:.6f}, {k4:.6f}]")

    # ---------------------------------------------------------------- #
    # 0. Where does the sensor actually live in ru?                      #
    # ---------------------------------------------------------------- #
    hr("0. Sensor extent in normalised radius ru")
    ru_halfw = (W / 2) / fx
    ru_halfh = (H / 2) / fy
    ru_corner = math.hypot((W / 2) / fx, (H / 2) / fy)
    print(f"ru at half-width  = {ru_halfw:.4f}")
    print(f"ru at half-height = {ru_halfh:.4f}")
    print(f"ru at corner      = {ru_corner:.4f}")

    # Monotonic range of the TRUE forward model
    th = torch.linspace(0.0, math.pi, 200001, dtype=torch.float64)
    ru_true = forward_ru(th, k)
    d1 = dru_dtheta(th, k)
    turn = torch.nonzero(d1 <= 0)
    if len(turn):
        theta_turn = th[turn[0, 0]].item()
        ru_turn = ru_true[turn[0, 0]].item()
        print(
            f"TRUE ru(theta) turns over (dru/dtheta<=0) at theta = {theta_turn:.4f} rad "
            f"({math.degrees(theta_turn):.2f} deg), ru = {ru_turn:.4f}"
        )
    else:
        theta_turn, ru_turn = math.pi, ru_true[-1].item()
        print("TRUE ru(theta) is monotonic over [0, pi]")

    # Lens limit = theta at the image corner, per the true model
    lo, hi = 0.0, theta_turn
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if forward_ru(torch.tensor(mid, dtype=torch.float64), k).item() < ru_corner:
            lo = mid
        else:
            hi = mid
    theta_corner = 0.5 * (lo + hi)
    print(f"theta at image corner (true model) = {theta_corner:.4f} rad " f"({math.degrees(theta_corner):.2f} deg)")

    # ---------------------------------------------------------------- #
    # 1. Round trip: theta -> ru (true) -> estimate_theta_star -> theta  #
    # ---------------------------------------------------------------- #
    hr("1. Round trip on the TRUE forward model, theta in [0, theta_corner]")
    theta_in = torch.linspace(0.0, theta_corner, 25, dtype=torch.float64)
    ru_in = forward_ru(theta_in, k)
    theta_out = estimate_theta_star(k1, k2, k3, k4, ru=ru_in, device=ru_in.device, dtype=ru_in.dtype)
    err = theta_out - theta_in
    print(f"{'theta_in':>10} {'deg':>8} {'ru':>10} {'theta_out':>11} {'err(rad)':>11} {'err(px)':>9}")
    for a, b, c, e in zip(theta_in, ru_in, theta_out, err):
        # px error at the sensor: dru = e * dru/dtheta, times fx
        px = abs(e.item() * dru_dtheta(a, k).item() * fx)
        print(
            f"{a.item():10.5f} {math.degrees(a.item()):8.2f} {b.item():10.5f} "
            f"{c.item():11.5f} {e.item():11.2e} {px:9.2f}"
        )
    max_rad = err.abs().max().item()
    max_px = max(abs(e.item() * dru_dtheta(a, k).item() * fx) for a, e in zip(theta_in, err))
    print(f"\nmax |err| = {max_rad:.4e} rad = {max_px:.4f} px")
    assert (
        max_rad < 1e-5 and max_px < 0.01
    ), f"round trip error {max_rad:.3e} rad / {max_px:.3f} px exceeds 1e-5 rad / 0.01 px"

    # ---------------------------------------------------------------- #
    # 2. Round trip against the polynomial AS CODED                      #
    # ---------------------------------------------------------------- #
    hr("2. Regression guard: ru built from the PRE-FIX (theta^2) polynomial")
    ru_bug = forward_ru_buggy(theta_in, k)
    theta_out2 = estimate_theta_star(k1, k2, k3, k4, ru=ru_bug, device=ru_bug.device, dtype=ru_bug.dtype)
    err2 = theta_out2 - theta_in
    print(f"max |err| = {err2.abs().max().item():.4e} rad")
    print("Near 0 => estimate_theta_star still inverts the buggy theta^2")
    print("polynomial (bug 4 present).  Large => it inverts true KB4 (bug 4 fixed).")
    assert err2.abs().max().item() > 1e-3, "estimate_theta_star still inverts the pre-fix theta^2 polynomial"

    # ---------------------------------------------------------------- #
    # 3. Units of the search range                                       #
    # ---------------------------------------------------------------- #
    hr("3. Units of the LUT search range")
    step = 0.001
    num_steps = int((math.pi - 0.0) / step)
    theta_vals = torch.linspace(0.0, math.pi, steps=num_steps, dtype=torch.float64)
    print(f"live code: linspace(0, pi, steps=int(pi/{step})) -> {num_steps} nodes")
    print(
        f"actual node spacing = {(theta_vals[1]-theta_vals[0]).item():.6f} rad "
        f"(step_size={step} is only used to set the COUNT, not the spacing)"
    )
    print(f"grid spans 0 .. {math.pi:.4f} rad = 0 .. 180 deg  -> RANGE IS RADIANS")
    print("the commented-out `theta_range = (0.0, 180.0)` is dead code and is")
    print("NOT used; if it were wired in as radians it would span 10313 deg.")
    # quantisation floor
    dtheta = (theta_vals[1] - theta_vals[0]).item()
    print(f"linear interpolation between nodes -> residual ~ (dtheta^2/8)*|d2ru/dtheta2|/dru/dtheta")
    print(f"node spacing {dtheta:.3e} rad = {dtheta*fx:.4f} px at CamA fx")

    # ---------------------------------------------------------------- #
    # 4. Behaviour past the lens limit                                   #
    # ---------------------------------------------------------------- #
    hr("4. Behaviour past the lens limit / past the LUT")
    ru_max_true = forward_ru(torch.tensor(theta_turn, dtype=torch.float64), k).item()
    ru_lut_max = forward_ru(torch.tensor(math.pi, dtype=torch.float64), k).item()
    print(f"TRUE   model: max invertible ru = {ru_max_true:.4f} at theta = {theta_turn:.4f} rad")
    print(f"AS-CODED LUT: R[-1] = {ru_lut_max:.4f} at theta = pi")
    probes = torch.tensor(
        [
            ru_corner,
            ru_max_true * 0.999,
            ru_max_true * 1.05,
            ru_lut_max * 0.999,
            ru_lut_max,
            ru_lut_max * 1.5,
            ru_lut_max * 10.0,
        ],
        dtype=torch.float64,
    )
    out = estimate_theta_star(k1, k2, k3, k4, ru=probes, device=probes.device, dtype=probes.dtype)
    print(f"\n{'ru':>12} {'theta_star':>12} {'deg':>10}   note")
    notes = [
        "image corner",
        "just below true turnover",
        "past true turnover",
        "just below LUT end",
        "at LUT end",
        "1.5x LUT end",
        "10x LUT end",
    ]
    for p, o, n in zip(probes, out, notes):
        print(f"{p.item():12.4f} {o.item():12.4f} {math.degrees(o.item()):10.2f}   {n}")
    print("\nru beyond R[-1] is clamped to the table, so theta_star saturates at pi.")
    assert (out <= math.pi + 1e-9).all(), "theta_star extrapolates past pi beyond the lookup table"

    # monotonicity of the AS-CODED table (searchsorted requires it)
    R_true = forward_ru(theta_vals, k)
    dR_true = R_true[1:] - R_true[:-1]
    n_neg_true = int((dR_true <= 0).sum())
    first = torch.nonzero(dR_true <= 0)
    print(
        f"\nLUT table monotonic? non-increasing steps = {n_neg_true} / {len(dR_true)}"
        + (f", first at theta = {theta_vals[first[0,0]].item():.4f} rad" if len(first) else "")
    )
    print("searchsorted() assumes a sorted table; a non-monotonic R silently")
    print("returns wrong indices for ru on the far branch.")

    # ---------------------------------------------------------------- #
    # 5. ru = 0 and negative ru                                          #
    # ---------------------------------------------------------------- #
    hr("5. Degenerate inputs")
    edge = torch.tensor([0.0, -0.1], dtype=torch.float64)
    o = estimate_theta_star(k1, k2, k3, k4, ru=edge, device=edge.device, dtype=edge.dtype)
    print(f"ru=0.0  -> theta_star={o[0].item():.6e}")
    print(f"ru=-0.1 -> theta_star={o[1].item():.6e}  (clamped to the table start)")
    print("note: generate_rays_kb4 divides by ru (m_x/ru) with no eps -> NaN if a")
    print("pixel lands exactly on the principal point.")

    # ---------------------------------------------------------------- #
    # 6. Size of the override, at CamA and CamD                          #
    # ---------------------------------------------------------------- #
    hr("6. Cost of the `theta_star = ru` override")
    for name, cid in (("CamA", 0), ("CamD", 3)):
        try:
            kk, (ffx, ffy, ccx, ccy), ww, hh = load_cam(cid)
        except KeyError:
            continue
        ru_c = math.hypot((ww / 2) / ffx, (hh / 2) / ffy)
        rt = torch.tensor([ru_c], dtype=torch.float64)
        th_true = estimate_theta_star(kk[0], kk[1], kk[2], kk[3], ru=rt, device=rt.device, dtype=rt.dtype)
        # true inversion by bisection on the correct polynomial
        lo, hi = 0.0, math.pi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if forward_ru(torch.tensor(mid, dtype=torch.float64), kk).item() < ru_c:
                lo = mid
            else:
                hi = mid
        th_bis = 0.5 * (lo + hi)
        print(f"{name}: corner ru={ru_c:.4f}  override theta=ru={ru_c:.4f} rad " f"({math.degrees(ru_c):.2f} deg)")
        print(
            f"      correct theta = {th_bis:.4f} rad ({math.degrees(th_bis):.2f} deg), "
            f"delta = {ru_c - th_bis:+.4f} rad"
        )


if __name__ == "__main__":
    main()
