"""Rendered frames -> MCAP images that vk_camera_driver can consume.

Written for the vio-sim-offline replay path (vk_playback ->
vk_camera_driver -> vk_vio). The driver copies each raw image's header +
intrinsic + extrinsic into the hfflow it emits, so ALL calibration must ride
in these messages. What this writer guarantees:

- header.stampMonotonic advances in real time (start_time_ns + index/fps) and
  is bit-identical across cameras for the same frame index; log_time =
  publish_time = stampMonotonic. (The old writer stamped seq*300 ms against
  33 ms log_time spacing - the 10x clock bug.)
- encoding mono8, step = actual row stride (= width), mipMapLevels 0.
- exposureUSec/gain > 0, streamName "camX/raw", frameId = device serial.
- intrinsic: EXACTLY ONE model group per camera, read from the device JSON
  given at construction. cameraType 1 -> kb4 (pinhole from intrinsicMatrix,
  k1..k4 = distortionCoeff[0:4]); cameraType 0 -> ds (fx,fy,cx,cy,xi,alpha =
  distortionCoeff[5:11], the same slice kaolin_future/fisheye.py renders
  with). Never touch the other groups: precedence is by capnp has-pointer.
- extrinsic.bodyFrame = body_T_cam and extrinsic.imuFrame = imu_T_cam (unit
  quaternions). Device JSON extrinsics are cam_i -> reference-socket with
  translation in CENTIMETRES; imuExtrinsics is imu -> reference-socket.
  body_T_reference = pure rotation BODY_Q_REF, the VK180-V2-FFC convention
  (vk180_tag_detection.json: body_T_cam0, reference_cam camd): optical
  x-right/y-down/z-forward -> NWU body x-forward/y-left/z-up.
- intrinsic/extrinsic lastModified = device JSON batchTime (UTC seconds,
  nonzero - 0 is fatal in vk_vio).
- channels registered as message_encoding "capnp::Image" on the canonical
  S1/camX topics (vk_playback whitelists topic names).

The IMU-lead rule (first imu_list stamp before the first image) is the bag
assembler's job; start_time_ns (default 1 s) leaves room for it.

CLI: re-encode an existing render bag (fixing all of the above) or convert
PNG folders:
    python mcap_convertor.py --calib calibration_files/DP180IP-30020104.json \
        --from-mcap orbit_4cam_kb4fix_img.mcap --output fixed.mcap [--fps 30]
"""

import argparse
import heapq
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
from mcap.writer import Writer

sys.path.append('/opt/vilota/messages')
import capnp
capnp.add_import_hook()
import image_capnp as eCALImage

CAMERA_NAMES = ["CamA", "CamB", "CamC", "CamD"]
CAM_SOCKET = {"CamA": 0, "CamB": 1, "CamC": 2, "CamD": 3}

# body_T_reference-camera rotation (x, y, z, w), zero translation.
# VK180-V2-FFC: reference_cam camd, body_T_cam0 = (.5, -.5, .5, -.5).
BODY_Q_REF = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float64)

DEFAULT_EXPOSURE_USEC = 2000  # small and honest: stamp = end of exposure
DEFAULT_GAIN = 400
DEFAULT_START_TIME_NS = 1_000_000_000  # room for imu_list to lead frame 0


def _quat_to_rot(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _rot_to_quat(R):
    """Rotation matrix -> unit quaternion (x, y, z, w), Shepperd's method."""
    m = np.asarray(R, dtype=np.float64)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _se3(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _inv_se3(T):
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def _check_rotation(R, what):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3) or np.abs(R @ R.T - np.eye(3)).max() > 1e-3 \
            or abs(np.linalg.det(R) - 1.0) > 1e-3:
        raise ValueError(f"{what}: not a proper rotation matrix: {R}")
    return R


class RigCalibration:
    """Per-camera intrinsics + body/imu extrinsics from a device JSON.

    Device JSON conventions (validated against DP180IP-30020104.json and the
    render code): cameraData entries are [socket, params]; extrinsics map
    points from this camera's frame to the toCameraSocket frame, translation
    in centimetres; the reference camera has toCameraSocket -1 and an empty
    rotationMatrix. imuExtrinsics maps imu-frame points to its toCameraSocket
    frame, centimetres as well.
    """

    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            data = json.load(f)

        self.serial = data.get('deviceName') or ''
        if not self.serial:
            self.serial = os.path.splitext(os.path.basename(path))[0]
            print(f"[mcap_convertor] WARNING: {path} has no deviceName; "
                  f"using frameId {self.serial!r}", file=sys.stderr)
        self.last_modified = int(data.get('batchTime') or 0)
        if self.last_modified == 0:
            # 0 is fatal downstream (vk_vio "missing calib" throw)
            self.last_modified = int(time.time())

        cam_entries = {sock: cam for sock, cam in data.get('cameraData', [])}
        if not cam_entries:
            raise ValueError(f"{path}: no cameraData")

        ref_sockets = [s for s, c in cam_entries.items()
                       if c['extrinsics']['toCameraSocket'] == -1
                       or not c['extrinsics']['rotationMatrix']]
        if len(ref_sockets) != 1:
            raise ValueError(f"{path}: expected exactly one reference camera,"
                             f" found sockets {ref_sockets}")
        self.reference_socket = ref_sockets[0]

        # ref_T_cam per socket (points: camera frame -> reference frame)
        ref_T_cam = {}
        for sock, cam in cam_entries.items():
            ext = cam['extrinsics']
            if sock == self.reference_socket:
                ref_T_cam[sock] = np.eye(4)
                continue
            if ext['toCameraSocket'] != self.reference_socket:
                raise ValueError(
                    f"{path}: socket {sock} chains to socket "
                    f"{ext['toCameraSocket']}, not the reference "
                    f"{self.reference_socket}; chained extrinsics unsupported")
            R = _check_rotation(ext['rotationMatrix'], f"socket {sock}")
            t_m = np.array([ext['translation'][k] for k in 'xyz']) / 100.0
            ref_T_cam[sock] = _se3(R, t_m)

        imu_ext = data.get('imuExtrinsics') or {}
        if not imu_ext.get('rotationMatrix'):
            raise ValueError(f"{path}: imuExtrinsics missing/empty - cannot "
                             "build extrinsic.imuFrame without it")
        if imu_ext['toCameraSocket'] != self.reference_socket:
            raise ValueError(f"{path}: imuExtrinsics.toCameraSocket "
                             f"{imu_ext['toCameraSocket']} != reference "
                             f"{self.reference_socket}")
        R_imu = _check_rotation(imu_ext['rotationMatrix'], "imuExtrinsics")
        t_imu = np.array([imu_ext['translation'][k] for k in 'xyz']) / 100.0
        imu_T_ref = _inv_se3(_se3(R_imu, t_imu))  # invert imu -> reference

        body_T_ref = _se3(_quat_to_rot(BODY_Q_REF), np.zeros(3))

        self.cams = {}
        for sock, cam in cam_entries.items():
            dc = cam['distortionCoeff']
            K = cam['intrinsicMatrix']
            if cam['cameraType'] == 1:
                model = ('kb4', dict(fx=K[0][0], fy=K[1][1], cx=K[0][2],
                                     cy=K[1][2], k1=dc[0], k2=dc[1],
                                     k3=dc[2], k4=dc[3]))
            elif cam['cameraType'] == 0:
                model = ('ds', dict(fx=dc[5], fy=dc[6], cx=dc[7], cy=dc[8],
                                    xi=dc[9], alpha=dc[10]))
            else:
                raise ValueError(f"{path}: socket {sock} has unknown "
                                 f"cameraType {cam['cameraType']}")
            self.cams[sock] = {
                'model': model[0],
                'params': model[1],
                'width': cam['width'],
                'height': cam['height'],
                'body_T_cam': body_T_ref @ ref_T_cam[sock],
                'imu_T_cam': imu_T_ref @ ref_T_cam[sock],
            }

    def for_cam_name(self, cam_name):
        sock = CAM_SOCKET[cam_name]
        if sock not in self.cams:
            raise ValueError(f"{self.path}: no calibration for {cam_name} "
                             f"(socket {sock}); sockets: {list(self.cams)}")
        return sock, self.cams[sock]


class McapConverter:
    def __init__(self, calibration_file=None, fps=None,
                 start_time_ns=DEFAULT_START_TIME_NS,
                 exposure_usec=DEFAULT_EXPOSURE_USEC, gain=DEFAULT_GAIN):
        self.output_filename = "output.mcap"
        self.output_folder = "./mcap_outputs/"
        # PLAYGROUND_FPS keeps the export stamps in step with a trajectory
        # generated at a non-default rate (vio_trajectory sets it).
        self.fps = float(fps if fps is not None
                         else os.environ.get("PLAYGROUND_FPS", 30.0))
        self.start_time_ns = int(start_time_ns)
        self.exposure_usec = int(exposure_usec)
        self.gain = int(gain)
        calibration_file = calibration_file or os.environ.get(
            'PLAYGROUND_CALIB')
        self.calibration = (RigCalibration(calibration_file)
                            if calibration_file else None)
        self._warned_no_calib = False
        self.set_filepath()

    def set_filepath(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.output_fullpath = os.path.join(self.output_folder,
                                            self.output_filename)
        print(f"Output file will be saved to: {self.output_fullpath}")

    def calculate_time_interval(self, fps=None):
        return int(round(1e9 / (fps or self.fps)))

    def _to_mono8(self, img_array):
        img = np.asarray(img_array)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame, got {img.dtype}")
        return np.ascontiguousarray(img)

    def build_image_message(self, img_array, index, cam_name, stamp_ns):
        """Build one section-2b-compliant vkc::Image from a frame.

        Args:
            img_array: (H, W) mono8 or (H, W, 3) BGR uint8 frame.
            index: frame index within the sequence (header.seq).
            cam_name: one of CAMERA_NAMES.
            stamp_ns: header.stampMonotonic; the caller must pass the SAME
                value for all cameras of the same frame (bit-identical stamps
                are what the driver's stereo sync matches on).
        """
        gray = self._to_mono8(img_array)
        h, w = gray.shape

        msg = eCALImage.Image.new_message()
        msg.header.seq = index
        msg.header.stampMonotonic = int(stamp_ns)
        msg.encoding = eCALImage.Image.Encoding.mono8
        msg.width, msg.height = w, h
        msg.step = w  # actual row stride; gray is contiguous mono8
        msg.data = gray.tobytes()
        msg.exposureUSec = self.exposure_usec
        msg.gain = self.gain
        msg.sensorIdx = CAM_SOCKET[cam_name]
        msg.streamName = f"{cam_name.lower()}/raw"
        if w % 8:
            print(f"[mcap_convertor] WARNING: width {w} not divisible by 8; "
                  "the driver's mipmap stage will throw for levels >= 2",
                  file=sys.stderr)

        if self.calibration is None:
            if not self._warned_no_calib:
                self._warned_no_calib = True
                print("[mcap_convertor] WARNING: no calibration file given "
                      "(constructor arg or PLAYGROUND_CALIB env). Writing "
                      "images WITHOUT intrinsic/extrinsic: vk_camera_driver "
                      "will silently emit zero-point hfflow from this bag.",
                      file=sys.stderr)
            return msg

        msg.header.frameId = self.calibration.serial
        _, cam = self.calibration.for_cam_name(cam_name)
        if (w, h) != (cam['width'], cam['height']):
            raise ValueError(
                f"{cam_name}: frame is {w}x{h} but the device file says "
                f"{cam['width']}x{cam['height']} - embedded intrinsics must "
                "match the rendered geometry")

        p = cam['params']
        if cam['model'] == 'kb4':
            # touch ONLY this group: model precedence is by capnp has-pointer
            kb4 = msg.intrinsic.kb4
            kb4.pinhole.fx, kb4.pinhole.fy = p['fx'], p['fy']
            kb4.pinhole.cx, kb4.pinhole.cy = p['cx'], p['cy']
            kb4.k1, kb4.k2, kb4.k3, kb4.k4 = p['k1'], p['k2'], p['k3'], p['k4']
        else:
            ds = msg.intrinsic.ds
            ds.pinhole.fx, ds.pinhole.fy = p['fx'], p['fy']
            ds.pinhole.cx, ds.pinhole.cy = p['cx'], p['cy']
            ds.xi, ds.alpha = p['xi'], p['alpha']
        msg.intrinsic.lastModified = self.calibration.last_modified

        for field, T in (('bodyFrame', cam['body_T_cam']),
                         ('imuFrame', cam['imu_T_cam'])):
            se3 = getattr(msg.extrinsic, field)
            se3.position.x, se3.position.y, se3.position.z = (
                float(v) for v in T[:3, 3])
            q = _rot_to_quat(T[:3, :3])
            se3.orientation.x, se3.orientation.y = float(q[0]), float(q[1])
            se3.orientation.z, se3.orientation.w = float(q[2]), float(q[3])
        msg.extrinsic.lastModified = self.calibration.last_modified
        return msg

    def write_cam_frame_to_mcap(self, writer, channels, cam_name, frame,
                                index, timestamp):
        """Append one frame; timestamp is ns since frame 0 (index/fps).

        stampMonotonic = log_time = publish_time = start_time_ns + timestamp,
        so stamps advance at exactly the playback pacing (this is the fix for
        the old 10x clock skew).
        """
        if cam_name not in channels:
            channels[cam_name] = writer.register_channel(
                schema_id=0,
                topic=f"S1/{cam_name.lower()}",
                message_encoding="capnp::Image",
                metadata={"camera_name": cam_name},
            )
        stamp = self.start_time_ns + int(timestamp)
        msg = self.build_image_message(frame, index, cam_name, stamp)
        writer.add_message(
            channel_id=channels[cam_name],
            log_time=stamp,
            publish_time=stamp,
            data=msg.to_bytes(),
        )

    def load_frame_from_png(self, index, folder_path):
        filename = os.path.join(folder_path, f"{index:03d}.png")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No image found at {filename}")
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    def get_num_frames(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"No folder at {folder_path}")
        return len(os.listdir(folder_path))


def _decode_to_gray(msg):
    """Luma plane from an existing bag's Image message."""
    enc = str(msg.encoding)
    w, h = msg.width, msg.height
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ('mono8', 'yuv420', 'nv12'):
        return data[:h * w].reshape(h, w)  # first plane is the luma
    if enc == 'bgr8':
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"cannot re-encode from encoding {enc}")


def _iter_topic(path, topic, max_frames):
    """Yield (seq, cam_name, gray) for one S1/camX topic of an existing bag."""
    from mcap.reader import make_reader
    cam_name = 'Cam' + topic.split('/')[-1][-1].upper()  # S1/camb -> CamB
    with open(path, 'rb') as f:
        it = make_reader(f).iter_messages(topics=[topic])
        if max_frames:
            it = itertools.islice(it, max_frames)
        for _, _, message in it:
            with eCALImage.Image.from_bytes(message.data) as msg:
                yield int(msg.header.seq), cam_name, _decode_to_gray(msg)


def reconvert_mcap(conv, in_path, out_path, cams=None, max_frames=None):
    """Re-encode an existing render bag with compliant stamps + calibration.

    Frames are grouped by header.seq (per-camera frame index in the old
    writer) and written time-ordered via a k-way merge, one open reader per
    topic, so a multi-GB bag never sits in memory.
    """
    from mcap.reader import make_reader
    with open(in_path, 'rb') as f:
        summary = make_reader(f).get_summary()
    if summary is None:
        raise ValueError(f"{in_path}: no mcap summary section")
    topics = sorted(ch.topic for ch in summary.channels.values()
                    if ch.topic.startswith('S1/cam'))
    if cams:
        topics = [t for t in topics
                  if 'Cam' + t[-1].upper() in cams]
    if not topics:
        raise ValueError(f"{in_path}: no S1/camX topics found")
    print(f"Re-encoding {topics} from {in_path}")

    interval = conv.calculate_time_interval()
    iters = [_iter_topic(in_path, t, max_frames) for t in topics]
    merged = heapq.merge(*iters, key=lambda item: (item[0], item[1]))
    n = 0
    with open(out_path, 'wb') as stream:
        writer = Writer(stream)
        writer.start(profile="VisualKit")
        channels = {}
        for seq, cam_name, gray in merged:
            conv.write_cam_frame_to_mcap(writer, channels, cam_name, gray,
                                         seq, seq * interval)
            n += 1
            if n % 200 == 0:
                print(f"  {n} messages written", flush=True)
        writer.finish()
    print(f"Wrote {n} messages to {out_path}")


def convert_dirs(conv, root, out_path, cams=None, max_frames=None):
    """Convert cam_streams-style PNG folders (root/CamX/000.png ...)."""
    cams = cams or [c for c in CAMERA_NAMES
                    if os.path.isdir(os.path.join(root, c))]
    if not cams:
        raise ValueError(f"no CamX folders under {root}")
    interval = conv.calculate_time_interval()
    counts = {c: conv.get_num_frames(os.path.join(root, c)) for c in cams}
    num_frames = min(counts.values())
    if max_frames:
        num_frames = min(num_frames, max_frames)
    print(f"Converting {num_frames} frames for {cams} from {root}")
    with open(out_path, 'wb') as stream:
        writer = Writer(stream)
        writer.start(profile="VisualKit")
        channels = {}
        for i in range(num_frames):
            for cam_name in cams:
                frame = conv.load_frame_from_png(i, os.path.join(root,
                                                                 cam_name))
                conv.write_cam_frame_to_mcap(writer, channels, cam_name,
                                             frame, i, i * interval)
        writer.finish()
    print(f"Wrote {num_frames} frames/cam to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--calib', required=True,
                    help='device JSON for the unit that made the frames')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--from-mcap', help='existing render bag to re-encode')
    src.add_argument('--from-dirs', help='folder with CamX/ PNG subfolders')
    ap.add_argument('--output', required=True)
    ap.add_argument('--fps', type=float, default=None,
                    help='frame rate (default: PLAYGROUND_FPS env, else 30)')
    ap.add_argument('--start-time', type=float, default=1.0,
                    help='stamp of frame 0 in seconds (imu_list must lead it)')
    ap.add_argument('--cams', nargs='*',
                    help='subset of CamA..CamD (default: all found)')
    ap.add_argument('--max-frames', type=int)
    args = ap.parse_args()

    conv = McapConverter(calibration_file=args.calib, fps=args.fps,
                         start_time_ns=int(args.start_time * 1e9))
    if args.from_mcap:
        reconvert_mcap(conv, args.from_mcap, args.output,
                       cams=args.cams, max_frames=args.max_frames)
    else:
        convert_dirs(conv, args.from_dirs, args.output,
                     cams=args.cams, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
