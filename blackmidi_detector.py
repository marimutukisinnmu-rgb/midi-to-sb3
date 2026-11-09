
import sys
import os
import time
import heapq
import struct
from collections import defaultdict

LIMIT = 300


class MIDIError(Exception):
    pass


# ==========================================
# Binary MIDI Reader
# ==========================================

def read_vlq(f, end):
    value = 0

    for _ in range(4):
        if f.tell() >= end:
            raise MIDIError("Unexpected end of track")

        data = f.read(1)
        if not data:
            raise MIDIError("Unexpected EOF")

        b = data[0]
        value = (value << 7) | (b & 0x7F)

        if not (b & 0x80):
            return value

    raise MIDIError("Invalid VLQ")


def read_byte(f, end):
    if f.tell() >= end:
        raise MIDIError("Unexpected end of track")

    data = f.read(1)

    if not data:
        raise MIDIError("Unexpected EOF")

    return data[0]


def skip_bytes(f, length, end):
    if length < 0 or f.tell() + length > end:
        raise MIDIError("Invalid event length")

    f.seek(length, 1)


def track_events(path, offset, length):
    """1トラックを逐次解析し、Noteイベントだけを出力する。"""

    end = offset + length

    with open(path, "rb") as f:
        f.seek(offset)

        tick = 0
        running_status = None

        pending_tick = None
        pending = []

        def flush():
            nonlocal pending

            result = [
                (pending_tick, priority, channel, note, kind)
                for priority, channel, note, kind in pending
            ]

            pending = []
            return result

        while f.tell() < end:
            delta = read_vlq(f, end)
            tick += delta

            first = read_byte(f, end)

            if first >= 0x80:
                status = first
                first_data = None
            else:
                if running_status is None:
                    raise MIDIError("Invalid running status")

                status = running_status
                first_data = first

            # Meta Event
            if status == 0xFF:
                running_status = None

                meta_type = read_byte(f, end)
                size = read_vlq(f, end)
                skip_bytes(f, size, end)

                if meta_type == 0x2F:
                    break

                continue

            # SysEx Event
            if status in (0xF0, 0xF7):
                running_status = None

                size = read_vlq(f, end)
                skip_bytes(f, size, end)
                continue

            # Channel Event
            if 0x80 <= status <= 0xEF:
                running_status = status

                event_type = status & 0xF0
                channel = status & 0x0F

                data_length = (
                    1 if event_type in (0xC0, 0xD0)
                    else 2
                )

                data1 = (
                    first_data
                    if first_data is not None
                    else read_byte(f, end)
                )

                data2 = (
                    read_byte(f, end)
                    if data_length == 2
                    else 0
                )

                if event_type == 0x90 and data2 > 0:
                    kind = 1
                elif event_type == 0x80 or (
                    event_type == 0x90 and data2 == 0
                ):
                    kind = -1
                else:
                    continue

                # Tickが変わったら前のTickを出力
                if pending_tick is not None and tick != pending_tick:
                    yield from flush()

                pending_tick = tick

                # Note Offを先に処理
                priority = 0 if kind == -1 else 1

                pending.append((
                    priority,
                    channel,
                    data1,
                    kind
                ))

                continue

            # その他のSystem Event
            running_status = None

            system_lengths = {
                0xF1: 1,
                0xF2: 2,
                0xF3: 1,
                0xF6: 0,
            }

            if status not in system_lengths:
                raise MIDIError(
                    f"Unsupported status: 0x{status:02X}"
                )

            skip_bytes(f, system_lengths[status], end)

        yield from flush()


# ==========================================
# MIDI Header / Track Scanner
# ==========================================

def read_midi_header(path):
    tracks = []

    with open(path, "rb") as f:
        if f.read(4) != b"MThd":
            raise MIDIError("Invalid MIDI header")

        raw = f.read(4)
        if len(raw) != 4:
            raise MIDIError("Invalid MIDI header")

        header_length = struct.unpack(">I", raw)[0]

        if header_length < 6:
            raise MIDIError("Invalid header length")

        raw = f.read(6)
        if len(raw) != 6:
            raise MIDIError("Incomplete MIDI header")

        midi_type, track_count, division = struct.unpack(
            ">HHH", raw
        )

        if midi_type not in (0, 1, 2):
            raise MIDIError("Unsupported MIDI format")

        if division == 0 or division & 0x8000:
            raise MIDIError("Unsupported time division")

        f.seek(8 + header_length)

        file_size = os.path.getsize(path)

        for _ in range(track_count):
            chunk_header = f.read(8)

            if len(chunk_header) != 8:
                raise MIDIError("Missing track chunk")

            chunk_type = chunk_header[:4]
            chunk_length = struct.unpack(
                ">I", chunk_header[4:]
            )[0]

            offset = f.tell()

            if offset + chunk_length > file_size:
                raise MIDIError("Track exceeds file size")

            if chunk_type == b"MTrk":
                tracks.append((offset, chunk_length))

            f.seek(offset + chunk_length)

    if len(tracks) != track_count:
        raise MIDIError("Track count mismatch")

    return midi_type, tracks


# ==========================================
# Polyphony Analyzer
# ==========================================

def analyze_stream(streams):
    merged = heapq.merge(
        *streams,
        key=lambda e: (e[0], e[1])
    )

    active = defaultdict(int)

    current = 0
    maximum = 0

    for tick, priority, channel, note, kind in merged:
        key = (channel, note)

        if kind == -1:
            if active[key] > 0:
                active[key] -= 1
                current -= 1
        else:
            active[key] += 1
            current += 1

            if current > maximum:
                maximum = current

            if maximum >= LIMIT:
                return maximum, False

    return maximum, True


def check_polyphony(path):
    midi_type, tracks = read_midi_header(path)

    if midi_type == 2:
        maximum = 0

        for offset, length in tracks:
            stream = track_events(path, offset, length)
            value, passed = analyze_stream([stream])

            maximum = max(maximum, value)

            if not passed:
                return maximum, False

        return maximum, True

    streams = [
        track_events(path, offset, length)
        for offset, length in tracks
    ]

    return analyze_stream(streams)


# ==========================================
# Main
# ==========================================

def main():
    if len(sys.argv) < 2:
        filename = os.path.basename(__file__)
        print(f"Usage: python {filename} file.mid")
        return

    path = sys.argv[1]

    if not os.path.isfile(path):
        print("[ERROR] File not found")
        return

    print("MIDI Polyphony Detector")
    print("-----------------------")
    print(f"File: {os.path.basename(path)}")
    print(f"Size: {os.path.getsize(path):,} bytes")
    print(f"Limit: {LIMIT}")
    print("Analyzing...")

    start = time.perf_counter()

    try:
        maximum, passed = check_polyphony(path)

        elapsed = time.perf_counter() - start

        print()
        print(f"Max Polyphony: {maximum}")

        if passed:
            print(f"[PASS] Polyphony below {LIMIT}")
        else:
            print(f"[FAIL] Polyphony reached {LIMIT}+")

        print(f"Processing Time: {elapsed:.3f}s")

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
