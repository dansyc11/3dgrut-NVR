"""Read back a render image bag and check every vio_offline CHECKLIST.md
section-2b requirement (raw images as vk_camera_driver input).

    python verify_mcap_2b.py --mcap out.mcap \
        --calib calibration_files/DP180IP-30020104.json [--fps 30]

Independent of mcap_convertor: expectations are re-derived here from the
checklist and the device JSON (quaternions are checked by rebuilding rotation
matrices, so the writer's matrix->quat conversion is cross-validated).
Exits nonzero if any check fails.
"""

import argparse
import json
import sys

import numpy as np

sys.path.append('/opt/vilota/messages')
import capnp
capnp.add_import_hook()
import image_capnp as eCALImage
from mcap.reader import make_reader

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def quat_to_rot(x, y, z, w):
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = np.array([x, y, z, w]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def se3_from_reader(se3):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(se3.orientation.x, se3.orientation.y,
                            se3.orientation.z, se3.orientation.w)
    T[:3, 3] = [se3.position.x, se3.position.y, se3.position.z]
    return T


def load_device_json(path):
    with open(path) as f:
        data = json.load(f)
    cams = {sock: cam for sock, cam in data['cameraData']}
    ref = [s for s, c in cams.items()
           if c['extrinsics']['toCameraSocket'] == -1
           or not c['extrinsics']['rotationMatrix']][0]
    ref_T_cam = {}
    for sock, cam in cams.items():
        if sock == ref:
            ref_T_cam[sock] = np.eye(4)
        else:
            T = np.eye(4)
            T[:3, :3] = np.array(cam['extrinsics']['rotationMatrix'])
            T[:3, 3] = [cam['extrinsics']['translation'][k] / 100.0
                        for k in 'xyz']  # device file is in centimetres
            ref_T_cam[sock] = T
    imu = data['imuExtrinsics']
    ref_T_imu = np.eye(4)
    ref_T_imu[:3, :3] = np.array(imu['rotationMatrix'])
    ref_T_imu[:3, 3] = [imu['translation'][k] / 100.0 for k in 'xyz']
    return data, cams, ref, ref_T_cam, ref_T_imu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mcap', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--fps', type=float, default=30.0)
    args = ap.parse_args()

    data, jcams, ref_sock, ref_T_cam, ref_T_imu = load_device_json(args.calib)
    serial = data.get('deviceName', '')
    socket_of = {'cama': 0, 'camb': 1, 'camc': 2, 'camd': 3}
    interval = round(1e9 / args.fps)

    per_topic = {}  # topic -> dict of collected series
    channel_encodings = {}
    with open(args.mcap, 'rb') as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            t = channel.topic
            channel_encodings[t] = channel.message_encoding
            rec = per_topic.setdefault(t, {
                'seq': [], 'stamp': [], 'log': [], 'pub': [], 'first': None})
            with eCALImage.Image.from_bytes(message.data) as msg:
                rec['seq'].append(int(msg.header.seq))
                rec['stamp'].append(int(msg.header.stampMonotonic))
                rec['log'].append(message.log_time)
                rec['pub'].append(message.publish_time)
                if rec['first'] is None:
                    rec['first'] = {
                        'frameId': str(msg.header.frameId),
                        'encoding': str(msg.encoding),
                        'width': msg.width, 'height': msg.height,
                        'step': msg.step, 'datalen': len(msg.data),
                        'mipMapLevels': msg.mipMapLevels,
                        'exposureUSec': msg.exposureUSec, 'gain': msg.gain,
                        'streamName': str(msg.streamName),
                        'has_pinhole': msg.intrinsic._has('pinhole'),
                        'has_ds': msg.intrinsic._has('ds'),
                        'has_kb4': msg.intrinsic._has('kb4'),
                        'has_radtan8': msg.intrinsic._has('radtan8'),
                        'intr_lastMod': msg.intrinsic.lastModified,
                        'ds': (msg.intrinsic.ds.to_dict()
                               if msg.intrinsic._has('ds') else None),
                        'kb4': (msg.intrinsic.kb4.to_dict()
                                if msg.intrinsic._has('kb4') else None),
                        'ext_lastMod': msg.extrinsic.lastModified,
                        'body_T_cam': se3_from_reader(msg.extrinsic.bodyFrame),
                        'imu_T_cam': se3_from_reader(msg.extrinsic.imuFrame),
                        'body_q_norm': np.linalg.norm([
                            msg.extrinsic.bodyFrame.orientation.x,
                            msg.extrinsic.bodyFrame.orientation.y,
                            msg.extrinsic.bodyFrame.orientation.z,
                            msg.extrinsic.bodyFrame.orientation.w]),
                        'imu_q_norm': np.linalg.norm([
                            msg.extrinsic.imuFrame.orientation.x,
                            msg.extrinsic.imuFrame.orientation.y,
                            msg.extrinsic.imuFrame.orientation.z,
                            msg.extrinsic.imuFrame.orientation.w]),
                        'pix_std': float(np.frombuffer(
                            msg.data, dtype=np.uint8).std()),
                    }

    # -- channel level ------------------------------------------------------
    topics = sorted(per_topic)
    print(f"Topics found: {topics}")
    for t in ('S1/camb', 'S1/camc', 'S1/camd'):
        check(f"{t} present (vk_playback name whitelist)", t in per_topic)
    for t in topics:
        check(f"{t} message_encoding capnp::Image",
              channel_encodings[t] == "capnp::Image",
              repr(channel_encodings[t]))

    # -- clocks -------------------------------------------------------------
    stamps_by_seq = {}
    for t, rec in per_topic.items():
        seq = np.array(rec['seq'])
        stamp = np.array(rec['stamp'])
        check(f"{t} header.seq monotonic +1",
              np.all(np.diff(seq) == 1) and seq[0] == 0,
              f"first {seq[0]}, diffs {set(np.diff(seq).tolist())}")
        check(f"{t} stampMonotonic strictly increasing, no duplicates",
              np.all(np.diff(stamp) > 0))
        check(f"{t} stamp == log_time == publish_time",
              rec['stamp'] == rec['log'] == rec['pub'])
        periods = set(np.diff(stamp).tolist())
        check(f"{t} frame period == 1/fps (10x clock bug gone)",
              periods == {interval},
              f"expected {interval} ns, got {sorted(periods)[:3]}")
        check(f"{t} first stamp > 0 (room for imu_list to lead)",
              stamp[0] > 0, f"first stamp {stamp[0]}")
        for s, st in zip(rec['seq'], rec['stamp']):
            stamps_by_seq.setdefault(s, set()).add(st)
    check("stamps bit-identical across cameras per frame",
          all(len(v) == 1 for v in stamps_by_seq.values()),
          f"{sum(len(v) != 1 for v in stamps_by_seq.values())} frames differ")

    # -- per-camera image struct -------------------------------------------
    for t in topics:
        m = per_topic[t]['first']
        sock = socket_of[t.split('/')[-1]]
        jc = jcams[sock]
        dc = jc['distortionCoeff']
        K = jc['intrinsicMatrix']

        check(f"{t} frameId == device serial",
              m['frameId'] == serial, repr(m['frameId']))
        check(f"{t} encoding mono8", m['encoding'] == 'mono8', m['encoding'])
        check(f"{t} width/height match device file",
              (m['width'], m['height']) == (jc['width'], jc['height']),
              f"{m['width']}x{m['height']}")
        check(f"{t} step == row stride == width",
              m['step'] == m['width'], f"step {m['step']}")
        check(f"{t} data length == step*height",
              m['datalen'] == m['step'] * m['height'])
        check(f"{t} mipMapLevels == 0 on raw", m['mipMapLevels'] == 0)
        check(f"{t} width % 8 == 0 (mipmap divisibility)",
              m['width'] % 8 == 0)
        check(f"{t} exposureUSec > 0", m['exposureUSec'] > 0,
              str(m['exposureUSec']))
        check(f"{t} gain > 0", m['gain'] > 0, str(m['gain']))
        check(f"{t} streamName set", m['streamName'] != '',
              repr(m['streamName']))
        check(f"{t} image data not blank", m['pix_std'] > 1.0,
              f"pixel std {m['pix_std']:.1f}")

        n_groups = sum([m['has_pinhole'], m['has_ds'], m['has_kb4'],
                        m['has_radtan8']])
        expect = 'ds' if jc['cameraType'] == 0 else 'kb4'
        check(f"{t} exactly ONE intrinsic model group", n_groups == 1,
              f"pinhole={m['has_pinhole']} ds={m['has_ds']} "
              f"kb4={m['has_kb4']} radtan8={m['has_radtan8']}")
        check(f"{t} model group is {expect} (cameraType {jc['cameraType']})",
              m[expect] is not None and n_groups == 1)
        if expect == 'kb4' and m['kb4']:
            got = m['kb4']
            want = dict(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2],
                        k1=dc[0], k2=dc[1], k3=dc[2], k4=dc[3])
            ok = (np.allclose(
                [got['pinhole'][k] for k in ('fx', 'fy', 'cx', 'cy')],
                [want[k] for k in ('fx', 'fy', 'cx', 'cy')], rtol=1e-6)
                and np.allclose([got[k] for k in ('k1', 'k2', 'k3', 'k4')],
                                [want[k] for k in ('k1', 'k2', 'k3', 'k4')],
                                rtol=1e-5, atol=1e-9))
            check(f"{t} kb4 values match device JSON exactly", ok,
                  f"{got} vs {want}")
        if expect == 'ds' and m['ds']:
            got = m['ds']
            want = dict(fx=dc[5], fy=dc[6], cx=dc[7], cy=dc[8],
                        xi=dc[9], alpha=dc[10])
            ok = (np.allclose(
                [got['pinhole'][k] for k in ('fx', 'fy', 'cx', 'cy')],
                [want[k] for k in ('fx', 'fy', 'cx', 'cy')], rtol=1e-6)
                and np.allclose([got['xi'], got['alpha']],
                                [want['xi'], want['alpha']], rtol=1e-5))
            check(f"{t} ds values match device JSON "
                  "(distortionCoeff[5:11], the render's slice)", ok,
                  f"{got} vs {want}")

        check(f"{t} intrinsic.lastModified nonzero (0 fatal in vk_vio)",
              m['intr_lastMod'] != 0, str(m['intr_lastMod']))
        check(f"{t} extrinsic.lastModified nonzero",
              m['ext_lastMod'] != 0, str(m['ext_lastMod']))
        check(f"{t} imuFrame quaternion is unit (zero quat -> NaN)",
              abs(m['imu_q_norm'] - 1.0) < 1e-9, f"norm {m['imu_q_norm']}")
        check(f"{t} bodyFrame quaternion is unit",
              abs(m['body_q_norm'] - 1.0) < 1e-9, f"norm {m['body_q_norm']}")

        # imu_T_cam must equal inv(ref_T_imu) @ ref_T_cam from the JSON
        want_imu = np.linalg.inv(ref_T_imu) @ ref_T_cam[sock]
        err = np.abs(m['imu_T_cam'] - want_imu).max()
        check(f"{t} imuFrame == imu_T_cam from device JSON", err < 1e-6,
              f"max abs err {err:.2e}")

    # -- cross-camera extrinsic geometry -----------------------------------
    # bodyFrame feeds the driver's essential matrices: relative transforms
    # cam_x -> cam_y must match the device JSON, and bodyFrame/imuFrame must
    # agree with each other (common rigid body_T_imu).
    firsts = {t: per_topic[t]['first'] for t in topics}
    for tx in topics:
        for ty in topics:
            if tx >= ty:
                continue
            sx, sy = socket_of[tx.split('/')[-1]], socket_of[ty.split('/')[-1]]
            want_rel = np.linalg.inv(ref_T_cam[sx]) @ ref_T_cam[sy]
            got_body = (np.linalg.inv(firsts[tx]['body_T_cam'])
                        @ firsts[ty]['body_T_cam'])
            got_imu = (np.linalg.inv(firsts[tx]['imu_T_cam'])
                       @ firsts[ty]['imu_T_cam'])
            eb = np.abs(got_body - want_rel).max()
            ei = np.abs(got_imu - want_rel).max()
            check(f"{tx}->{ty} relative pose from bodyFrame matches JSON",
                  eb < 1e-6, f"max abs err {eb:.2e}")
            check(f"{tx}->{ty} relative pose from imuFrame matches JSON",
                  ei < 1e-6, f"max abs err {ei:.2e}")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f_ in FAILURES:
            print(f"  - {f_}")
        sys.exit(1)
    print("ALL SECTION-2b CHECKS PASSED")


if __name__ == "__main__":
    main()
