#!/usr/bin/env python3
"""Relabel a playground MCAP so the Vilota tools accept it.

The frames are already capnp Image messages in yuv420.
Two fields are wrong:
  - the channel message encoding says image/jpeg
  - the step field is 0, but it must equal the row width

This script fixes both. It does not touch the pixels.

Usage:
    python fix_mcap_labels.py input.mcap output.mcap
    python fix_mcap_labels.py input.mcap output.mcap --topics S1/camd
"""

import argparse
import os
import sys

import capnp
from mcap.reader import make_reader
from mcap.writer import Writer

sys.path.append("/opt/vilota/messages")
capnp.add_import_hook()

import image_capnp as ImageSchema  # noqa: E402

SCHEMA_PATH = "/opt/vilota/messages/image.capnp"


def load_schema_bytes():
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "rb") as f:
            return f.read()
    return b""


def convert(in_path, out_path, wanted, serial=None):
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        reader = make_reader(fin)
        writer = Writer(fout)
        writer.start(profile="VisualKit", library="fix_mcap_labels")

        schema_id = writer.register_schema(
            name="vkc.Image", encoding="capnp", data=load_schema_bytes()
        )

        channels = {}
        counts = {}

        for schema, channel, message in reader.iter_messages():
            if wanted and channel.topic not in wanted:
                continue

            try:
                with ImageSchema.Image.from_bytes(message.data) as old:
                    new = ImageSchema.Image.new_message()
                    new.header.seq = old.header.seq
                    new.header.stampMonotonic = old.header.stampMonotonic
                    new.header.frameId = serial if serial else old.header.frameId
                    new.encoding = old.encoding
                    new.width = old.width
                    new.height = old.height
                    new.step = old.step if old.step else old.width
                    new.data = old.data
                    new.exposureUSec = old.exposureUSec
                    new.gain = old.gain
                    new.sensorIdx = old.sensorIdx
                    new.streamName = old.streamName
                    new.mipMapLevels = old.mipMapLevels
                    payload = new.to_bytes()
                    shape = (old.width, old.height, str(old.encoding))
            except Exception as e:
                print(f"skip {channel.topic}: {e}")
                continue

            if channel.topic not in channels:
                channels[channel.topic] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding="capnp::Image",
                    schema_id=schema_id,
                )
                print(f"channel {channel.topic}: {shape[0]}x{shape[1]} {shape[2]}")

            writer.add_message(
                channel_id=channels[channel.topic],
                log_time=message.log_time,
                publish_time=message.publish_time,
                data=payload,
            )
            counts[channel.topic] = counts.get(channel.topic, 0) + 1

        writer.finish()

    for topic, n in sorted(counts.items()):
        print(f"wrote {n} frames on {topic}")
    if serial:
        print(f"serial written into every header: {serial}")
    if not counts:
        print("wrote nothing: no channel matched")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--topics", nargs="*", default=None)
    p.add_argument("--serial", default=None,
                   help="Write this device serial into every header frameId. "
                        "vk_calibrate needs it, and the playground leaves it empty.")
    args = p.parse_args()
    convert(args.input, args.output,
            set(args.topics) if args.topics else None, args.serial)


if __name__ == "__main__":
    main()
