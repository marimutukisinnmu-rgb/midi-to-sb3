import os
import struct
from dataclasses import dataclass, field


class MIDIParseError(Exception):
    """MIDI解析エラー"""
    pass


@dataclass
class MIDIEvent:
    tick: int
    type: str
    channel: int = None
    note: int = None
    velocity: int = None
    program: int = None
    controller: int = None
    value: int = None
    tempo: int = None
    text: str = None


@dataclass
class MIDITrack:
    name: str = ""
    events: list = field(default_factory=list)


@dataclass
class MIDIFile:
    format: int
    ticks_per_beat: int
    tracks: list


def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise MIDIParseError("Unexpected end of file")
    return data


def read_byte(f, end):
    if f.tell() >= end:
        raise MIDIParseError("Unexpected end of track")
    return read_exact(f, 1)[0]


def read_vlq(f, end):
    value = 0
    for _ in range(4):
        byte = read_byte(f, end)
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value
    raise MIDIParseError("Invalid VLQ")


def skip_bytes(f, size, end):
    if size < 0 or f.tell() + size > end:
        raise MIDIParseError("Invalid event length")
    f.seek(size, 1)


def parse_track(path, offset, length):
    """Parse one MIDI track and retain supported events."""
    end = offset + length
    track = MIDITrack()

    with open(path, "rb") as f:
        f.seek(offset)
        tick = 0
        running_status = None

        while f.tell() < end:
            tick += read_vlq(f, end)
            first = read_byte(f, end)

            if first >= 0x80:
                status = first
                first_data = None
            else:
                if running_status is None:
                    raise MIDIParseError("Invalid running status")
                status = running_status
                first_data = first

            # Meta events
            if status == 0xFF:
                running_status = None
                meta_type = read_byte(f, end)
                size = read_vlq(f, end)

                if f.tell() + size > end:
                    raise MIDIParseError("Meta event exceeds track")
                data = read_exact(f, size)

                if meta_type == 0x03:
                    track.name = data.decode("utf-8", errors="replace")
                    track.events.append(
                        MIDIEvent(tick, "track_name", text=track.name)
                    )
                elif meta_type == 0x51:
                    if size != 3:
                        raise MIDIParseError("Invalid tempo event")
                    track.events.append(
                        MIDIEvent(tick, "tempo",
                                  tempo=int.from_bytes(data, "big"))
                    )
                elif meta_type == 0x2F:
                    break
                continue

            # SysEx events
            if status in (0xF0, 0xF7):
                running_status = None
                size = read_vlq(f, end)
                skip_bytes(f, size, end)
                continue

            # Channel voice events
            if 0x80 <= status <= 0xEF:
                running_status = status
                event_type = status & 0xF0
                channel = status & 0x0F
                data_length = 1 if event_type in (0xC0, 0xD0) else 2

                data1 = (
                    first_data if first_data is not None
                    else read_byte(f, end)
                )
                data2 = read_byte(f, end) if data_length == 2 else None

                if data1 >= 0x80 or (data2 is not None and data2 >= 0x80):
                    raise MIDIParseError("Invalid MIDI data byte")

                if event_type == 0x80:
                    track.events.append(
                        MIDIEvent(tick, "note_off", channel=channel,
                                  note=data1, velocity=data2)
                    )
                elif event_type == 0x90:
                    event_name = "note_off" if data2 == 0 else "note_on"
                    track.events.append(
                        MIDIEvent(tick, event_name, channel=channel,
                                  note=data1, velocity=data2)
                    )
                elif event_type == 0xC0:
                    track.events.append(
                        MIDIEvent(tick, "program_change", channel=channel,
                                  program=data1)
                    )
                elif event_type == 0xB0 and data1 == 7:
                    track.events.append(
                        MIDIEvent(tick, "volume", channel=channel,
                                  controller=data1, value=data2)
                    )
                continue

            # Other system common/realtime events
            running_status = None
            system_lengths = {0xF1: 1, 0xF2: 2, 0xF3: 1, 0xF6: 0}
            if status not in system_lengths:
                raise MIDIParseError(f"Unsupported status: 0x{status:02X}")
            skip_bytes(f, system_lengths[status], end)

    return track


def parse_midi(path):
    """Parse a Standard MIDI File into MIDIFile/MIDITrack/MIDIEvent objects."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    tracks_info = []

    with open(path, "rb") as f:
        if read_exact(f, 4) != b"MThd":
            raise MIDIParseError("Invalid MIDI header")

        header_length = struct.unpack(">I", read_exact(f, 4))[0]
        if header_length < 6:
            raise MIDIParseError("Invalid header length")

        midi_format, track_count, division = struct.unpack(
            ">HHH", read_exact(f, 6)
        )

        if midi_format not in (0, 1, 2):
            raise MIDIParseError("Unsupported MIDI format")
        if division == 0 or division & 0x8000:
            raise MIDIParseError("Unsupported time division")

        f.seek(8 + header_length)
        file_size = os.path.getsize(path)

        while len(tracks_info) < track_count:
            chunk_header = read_exact(f, 8)
            chunk_type = chunk_header[:4]
            chunk_length = struct.unpack(">I", chunk_header[4:])[0]
            offset = f.tell()

            if offset + chunk_length > file_size:
                raise MIDIParseError("Chunk exceeds file size")

            if chunk_type == b"MTrk":
                tracks_info.append((offset, chunk_length))
            f.seek(offset + chunk_length)

    tracks = [
        parse_track(path, offset, length)
        for offset, length in tracks_info
    ]

    if midi_format == 0 and len(tracks) != 1:
        raise MIDIParseError("Format 0 MIDI must contain exactly one track")

    return MIDIFile(
        format=midi_format,
        ticks_per_beat=division,
        tracks=tracks
    )
