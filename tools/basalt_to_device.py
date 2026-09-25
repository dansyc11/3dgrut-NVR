"""Basalt (vk_calibrate) calibration -> device JSON for mcap_convertor --calib.

    python tools/basalt_to_device.py calibration-kb4-kb4-kb4-ds.json \
        --template calibration_files/DP180IP-30020104.json --output fitted.json \
        [--compare calibration_files/DP180IP-30020104.json]

Camera order: the Basalt file names its cameras (value0.cam_names, S1/cama ..
S1/camd) and each camera goes to the socket its name gives (cama 0, camb 1,
camc 2, camd 3, as mcap_convertor CAM_SOCKET). The index table of the
production script /opt/vilota/bin/03-prepare_device_calibration_from_basalt.py
is NOT used: its VK180 branch maps index 0/1/2 to CAM_B/C/D, a three-camera
solve. Every camera's resolution must match the template's for its socket.

Extrinsics follow the production script: for every camera except the
template's reference camera (toCameraSocket -1), ref_T_cam =
inv(T_imu_ref) T_imu_cam, rotation as a nested 3x3, translation x100
(centimetres), specTranslation (-1, 0, 0), toCameraSocket = reference socket.
The reference camera keeps an empty rotationMatrix, zero translation and
toCameraSocket -1.

Intrinsics: kb4 -> cameraType 1, intrinsicMatrix from fx fy cx cy,
distortionCoeff[0:4] = k1..k4. ds -> cameraType 0, distortionCoeff[5:11] =
fx fy cx cy xi alpha. The device format also carries a KB4 fit for a DS camera
(intrinsicMatrix and distortionCoeff[0:4], filled by production's separate
all-KB4 solve). A mixed kb4/ds solve has no such fit, so those slots are
written as zeros instead of being guessed or copied from the template;
mcap_convertor and reproject_far never read them for cameraType 0.

Everything else (imuExtrinsics, housing, stereo rectification, board and
product info) is copied from the template: a camera-only solve (Basalt
T_imu_cam[0] = identity, empty imu_name) fits no IMU, and mcap_convertor needs
imuExtrinsics to write extrinsic.imuFrame. deviceName comes from the Basalt
serial_number and must equal the template's. batchTime is the Basalt file's
modification time, as in the production script.

--compare REF.json prints, per camera, the intrinsic errors and the
extrinsic rotation (deg) and position (mm) errors of the written file against
REF, both relative to the reference camera, and checks the camera order: each
camera's position must be nearest to its own socket's position in REF.
"""

import argparse
import copy
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot

SOCKET = {"cama": 0, "camb": 1, "camc": 2, "camd": 3}
LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}


def se3(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def basalt_pose(p):
    return se3(Rot.from_quat([p["qx"], p["qy"], p["qz"], p["qw"]]).as_matrix(), [p["px"], p["py"], p["pz"]])


def reference_socket(cams):
    refs = [s for s, c in cams.items() if c["extrinsics"]["toCameraSocket"] == -1]
    if len(refs) != 1:
        raise ValueError(f"expected exactly one reference camera, found sockets {refs}")
    return refs[0]


def convert(basalt_path, template_path):
    v = json.load(open(basalt_path))["value0"]
    tpl = json.load(open(template_path))
    tpl_cams = {int(s): c for s, c in tpl["cameraData"]}
    ref = reference_socket(tpl_cams)

    names = v["cam_names"]
    n = len(names)
    for key in ("intrinsics", "T_imu_cam", "resolution"):
        if len(v[key]) != n:
            raise ValueError(f"{basalt_path}: {n} cam_names but {len(v[key])} {key}")
    sockets = [SOCKET[name.split("/")[-1]] for name in names]
    if len(set(sockets)) != n:
        raise ValueError(f"{basalt_path}: cam_names {names} map to repeated sockets {sockets}")
    if ref not in sockets:
        raise ValueError(f"template reference socket {ref} is not in the solve {names}")

    print("camera order (Basalt index -> socket):")
    for i, (name, sock) in enumerate(zip(names, sockets)):
        w, h = v["resolution"][i]
        tc = tpl_cams.get(sock)
        if tc is None or (tc["width"], tc["height"]) != (w, h):
            raise ValueError(
                f"index {i} {name} -> socket {sock}: {w}x{h} does not match the "
                f"template's {None if tc is None else (tc['width'], tc['height'])}"
            )
        print(
            f"  {i}  {name:8} -> socket {sock} (Cam{LABEL[sock]})  "
            f"{v['intrinsics'][i]['camera_type']:3}  {w}x{h}  matches template"
        )

    T_imu = {sock: basalt_pose(v["T_imu_cam"][i]) for i, sock in enumerate(sockets)}
    T_imu_ref_inv = np.linalg.inv(T_imu[ref])

    out_cams = {}
    for i, sock in enumerate(sockets):
        intr = v["intrinsics"][i]
        p = intr["intrinsics"]
        dc = [0.0] * 14
        K = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        if intr["camera_type"] == "kb4":
            cam_type = 1
            K = [[p["fx"], 0.0, p["cx"]], [0.0, p["fy"], p["cy"]], [0.0, 0.0, 1.0]]
            dc[0:4] = [p["k1"], p["k2"], p["k3"], p["k4"]]
        elif intr["camera_type"] == "ds":
            cam_type = 0
            dc[5:11] = [p["fx"], p["fy"], p["cx"], p["cy"], p["xi"], p["alpha"]]
        else:
            raise ValueError(f"index {i}: unsupported camera_type {intr['camera_type']}")

        if sock == ref:
            ext = {
                "rotationMatrix": [],
                "specTranslation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "toCameraSocket": -1,
                "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
            }
        else:
            T = T_imu_ref_inv @ T_imu[sock]
            t_cm = T[:3, 3] * 100.0
            ext = {
                "rotationMatrix": T[:3, :3].tolist(),
                "specTranslation": {"x": -1.0, "y": 0.0, "z": 0.0},
                "toCameraSocket": ref,
                "translation": {"x": float(t_cm[0]), "y": float(t_cm[1]), "z": float(t_cm[2])},
            }

        tc = tpl_cams[sock]
        w, h = v["resolution"][i]
        out_cams[sock] = {
            "cameraType": cam_type,
            "distortionCoeff": [float(x) for x in dc],
            "extrinsics": ext,
            "height": int(h),
            "intrinsicMatrix": K,
            "lensPosition": tc.get("lensPosition", 0),
            "specHfovDeg": tc.get("specHfovDeg", 0.0),
            "width": int(w),
        }

    serial = v.get("serial_number", "")
    if serial != tpl.get("deviceName"):
        raise ValueError(f"Basalt serial_number {serial!r} != template deviceName " f"{tpl.get('deviceName')!r}")
    out = copy.deepcopy(tpl)
    order = [int(s) for s, _ in tpl["cameraData"] if int(s) in out_cams]
    out["cameraData"] = [[s, out_cams[s]] for s in order]
    out["deviceName"] = serial
    out["batchTime"] = int(os.path.getmtime(basalt_path))
    return out


def load_device(path_or_dict):
    """-> ({socket: (kind, params, T_ref_cam with t in m)}, reference socket)."""
    d = path_or_dict if isinstance(path_or_dict, dict) else json.load(open(path_or_dict))
    cams = {int(s): c for s, c in d["cameraData"]}
    ref = reference_socket(cams)
    res = {}
    for sock, c in cams.items():
        e = c["extrinsics"]
        R = np.array(e["rotationMatrix"], float) if len(e["rotationMatrix"]) else np.eye(3)
        t = np.array([e["translation"][k] for k in "xyz"], float) / 100.0
        dc = c["distortionCoeff"]
        if c["cameraType"] == 1:
            K = c["intrinsicMatrix"]
            params = dict(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2], k1=dc[0], k2=dc[1], k3=dc[2], k4=dc[3])
            kind = "kb4"
        else:
            params = dict(zip(("fx", "fy", "cx", "cy", "xi", "alpha"), dc[5:11]))
            kind = "ds"
        res[sock] = (kind, params, se3(R, t))
    return res, ref


def compare(fit, reference):
    f, ref_f = load_device(fit)
    g, ref_g = load_device(reference)
    if ref_f != ref_g:
        raise ValueError(f"reference cameras differ: {ref_f} vs {ref_g}")
    print(f"\nfitted vs reference, extrinsics relative to Cam{LABEL[ref_g]} (socket {ref_g})")
    hdr = (
        f"{'cam':3} {'model':5} {'fx %':>8} {'fy %':>8} {'cx px':>7} {'cy px':>7} | "
        f"{'distortion rel %':34} | {'rot deg':>8} {'pos mm':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for sock in sorted(g):
        if sock not in f:
            print(f"{LABEL[sock]:3} missing from the fitted file")
            continue
        kind, p, T = f[sock]
        kind_g, q, T_g = g[sock]
        if kind != kind_g:
            print(f"{LABEL[sock]:3} model differs: fitted {kind}, reference {kind_g}")
            continue
        keys = ("k1", "k2", "k3", "k4") if kind == "kb4" else ("xi", "alpha")
        dist = " ".join(f"{k} {100 * (p[k] - q[k]) / q[k]:+.3f}" for k in keys)
        # Device files store rotations in float32 (off-orthonormal by ~1e-7);
        # arccos((trace - 1) / 2) on them overstates angles below ~0.05 deg.
        rot = np.degrees(Rot.from_matrix(T_g[:3, :3].T @ T[:3, :3]).magnitude())
        pos = 1000.0 * np.linalg.norm(T[:3, 3] - T_g[:3, 3])
        print(
            f"{LABEL[sock]:3} {kind:5} {100 * (p['fx'] - q['fx']) / q['fx']:+8.4f} "
            f"{100 * (p['fy'] - q['fy']) / q['fy']:+8.4f} {p['cx'] - q['cx']:+7.3f} "
            f"{p['cy'] - q['cy']:+7.3f} | {dist:34} | "
            + ("       -       -" if sock == ref_g else f"{rot:8.4f} {pos:7.3f}")
        )

    print("\ncamera order check: fitted position vs every reference camera position [cm]")
    for sock in sorted(f):
        d = {s: 100.0 * np.linalg.norm(f[sock][2][:3, 3] - g[s][2][:3, 3]) for s in g}
        nearest = min(d, key=d.get)
        runner = sorted(d.values())[1]
        verdict = "own socket" if nearest == sock else f"WRONG: nearest is Cam{LABEL[nearest]}"
        print(
            f"  Cam{LABEL[sock]}: nearest reference camera Cam{LABEL[nearest]} at {d[nearest]:.3f} cm, "
            f"next {runner:.3f} cm -> {verdict}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basalt", help="vk_calibrate / Basalt calibration JSON")
    ap.add_argument(
        "--template", required=True, help="device JSON of the same unit: socket layout, reference camera, IMU block"
    )
    ap.add_argument("--output", required=True)
    ap.add_argument("--compare", default="", help="device JSON to diff the written file against")
    args = ap.parse_args()

    out = convert(args.basalt, args.template)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=4)
    print(
        f"wrote {args.output}  (deviceName {out['deviceName']}, batchTime {out['batchTime']}; "
        f"imuExtrinsics and non-camera fields copied from {os.path.basename(args.template)})"
    )
    if args.compare:
        compare(args.output, args.compare)


if __name__ == "__main__":
    main()
