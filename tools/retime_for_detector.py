"""Retime an image MCAP so vk_camera_driver's AprilGrid gate passes every frame.

    python tools/retime_for_detector.py input.mcap output.mcap [--step-ms 300]

Why: the driver schedules AprilGrid detection by the image header's
stampMonotonic, running at most once per ~250 ms of STREAM time (tag
detection "may run at a lower frequency", tagdetection.capnp:6). A bag
written by mcap_convertor carries real-time stamps (33 ms at 30 fps), so the
detector keeps exactly every 8th frame (8 x 33.3 = 266.7 ms, the first
multiple >= the gate) no matter how slowly vk_playback replays it - the gate
reads stamps, not arrival times, so playback rate and topic :queued modes
change nothing. The old writer stamped seq * 300 ms (a known bug), which is
why the office_v9 bag sailed through the gate at full rate.

This tool rewrites stampMonotonic to seq * step (default 300 ms, cloning the
proven v9 timing) on every image topic. MCAP log/publish times are kept, so
wall-clock playback pacing is unchanged (the detector keeps up at 15 fps
wall: v9 recorded all 1266 framesets in 84.6 s). Detections inherit the
synthetic stamps; frame identity stays header.seq, which is untouched.
"""

import argparse
import sys

import capnp
from mcap.reader import make_reader
from mcap.writer import Writer

sys.path.append("/opt/vilota/messages")
capnp.add_import_hook()
import image_capnp as ImageSchema  # noqa: E402

SCHEMA_PATH = "/opt/vilota/messages/image.capnp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument(
        "--step-ms",
        type=float,
        default=300.0,
        help="stampMonotonic spacing per seq (default 300 ms, " "the v9 timing; must be >= ~250 ms to pass the gate)",
    )
    args = ap.parse_args()
    step_ns = int(args.step_ms * 1e6)

    with open(SCHEMA_PATH, "rb") as f:
        schema_bytes = f.read()

    n = 0
    with open(args.input, "rb") as fin, open(args.output, "wb") as fout:
        reader = make_reader(fin)
        writer = Writer(fout)
        writer.start(profile="VisualKit", library="retime_for_detector")
        schema_id = writer.register_schema(name="vkc.Image", encoding="capnp", data=schema_bytes)
        channels = {}
        for schema, channel, message in reader.iter_messages():
            if channel.topic not in channels:
                channels[channel.topic] = writer.register_channel(
                    topic=channel.topic, message_encoding=channel.message_encoding, schema_id=schema_id
                )
            with ImageSchema.Image.from_bytes(message.data) as old:
                b = old.as_builder()
                b.header.stampMonotonic = int(b.header.seq) * step_ns
                data = b.to_bytes()
            writer.add_message(
                channel_id=channels[channel.topic],
                log_time=message.log_time,
                data=data,
                publish_time=message.publish_time,
            )
            n += 1
        writer.finish()
    print(
        f"retimed {n} messages ({len(channels)} topics) -> {args.output}, "
        f"stampMonotonic = seq * {args.step_ms:g} ms, log/publish times kept"
    )


if __name__ == "__main__":
    main()
