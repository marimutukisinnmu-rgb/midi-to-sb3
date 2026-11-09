"""Build Scratch 3.0 SB3 projects from parsed MIDI data.

The builder uses Scratch's standard Music extension. Each MIDI track becomes
one sprite. Notes are assigned to concurrent lanes (scripts) so overlapping
notes can play together.
"""

import hashlib
import json
import os
import heapq
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass


class SB3BuildError(Exception):
    """Raised when MIDI data cannot be represented as an SB3 project."""
    pass


# General MIDI program numbers (0-127) are approximated to Scratch Music
# extension instruments (1-21). Scratch has fewer instruments than GM.
GM_TO_SCRATCH_INSTRUMENT = (
    [1] * 8 +       # Piano
    [17] * 8 +      # Chromatic percussion -> Vibraphone
    [3] * 8 +       # Organ
    [4] * 8 +       # Guitar
    [6] * 8 +       # Bass
    [8] * 8 +       # Strings -> Cello
    [9] * 8 +       # Ensemble -> Choir
    [10] * 8 +      # Brass -> Trombone
    [11] * 8 +      # Reed -> Clarinet
    [12] * 8 +      # Pipe -> Flute
    [20] * 8 +      # Synth lead
    [21] * 8 +      # Synth pad
    [21] * 8 +      # Synth effects
    [18] * 8 +      # Ethnic -> Steel drum
    [17] * 8 +      # Percussive -> Vibraphone
    [21] * 8        # Sound effects -> Synth pad
)


@dataclass
class Note:
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int
    instrument: int


def _event_value(event, name, default=None):
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def _track_notes(track):
    """Pair Note On/Off events and remember the active GM program per channel."""
    active = defaultdict(deque)
    programs = defaultdict(int)
    notes = []
    events = sorted(
        enumerate(track.events),
        key=lambda item: (item[1].tick, item[0])
    )

    last_tick = 0

    for _, event in events:
        tick = int(_event_value(event, "tick", 0))
        last_tick = max(last_tick, tick)
        event_type = _event_value(event, "type")
        channel = _event_value(event, "channel", 0)

        if event_type == "program_change":
            program = int(_event_value(event, "program", 0))
            programs[channel] = max(0, min(127, program))

        elif event_type == "note_on":
            pitch = int(_event_value(event, "note", 0))
            velocity = int(_event_value(event, "velocity", 0))
            active[(channel, pitch)].append(
                (tick, velocity, programs[channel])
            )

        elif event_type == "note_off":
            pitch = int(_event_value(event, "note", 0))
            queue = active[(channel, pitch)]
            if queue:
                start, velocity, program = queue.popleft()
                if tick > start:
                    notes.append(
                        Note(
                            start=start,
                            end=tick,
                            pitch=pitch,
                            velocity=velocity,
                            channel=channel,
                            instrument=GM_TO_SCRATCH_INSTRUMENT[program],
                        )
                    )

    # Close notes that did not receive Note Off at the final track tick.
    for (channel, pitch), queue in active.items():
        while queue:
            start, velocity, program = queue.popleft()
            if last_tick > start:
                notes.append(
                    Note(
                        start=start,
                        end=last_tick,
                        pitch=pitch,
                        velocity=velocity,
                        channel=channel,
                        instrument=GM_TO_SCRATCH_INSTRUMENT[program],
                    )
                )

    return sorted(notes, key=lambda note: (note.start, note.end, note.pitch))


def _assign_lanes(notes):
    """Greedy interval partitioning: one lane per simultaneously active note."""
    lanes = []
    busy = []

    for note in notes:
        if busy and busy[0][0] <= note.start:
            _, lane_index = heapq.heappop(busy)
        else:
            lane_index = len(lanes)
            lanes.append([])

        lanes[lane_index].append(note)
        heapq.heappush(busy, (note.end, lane_index))

    return lanes


def _new_id(counter):
    counter[0] += 1
    return f"b{counter[0]:08x}"


def _number_input(value):
    return [1, [4, str(value)]]


def _make_chain(blocks, counter, specs, x, y):
    """Create a green-flag hat and a linear chain of extension blocks."""
    hat_id = _new_id(counter)
    block_ids = [_new_id(counter) for _ in specs]

    blocks[hat_id] = {
        "opcode": "event_whenflagclicked",
        "next": block_ids[0] if block_ids else None,
        "parent": None,
        "inputs": {},
        "fields": {},
        "shadow": False,
        "topLevel": True,
        "x": x,
        "y": y,
    }

    for index, spec in enumerate(specs):
        block_id = block_ids[index]
        parent_id = hat_id if index == 0 else block_ids[index - 1]
        next_id = block_ids[index + 1] if index + 1 < len(block_ids) else None

        blocks[block_id] = {
            "opcode": spec["opcode"],
            "next": next_id,
            "parent": parent_id,
            "inputs": spec.get("inputs", {}),
            "fields": spec.get("fields", {}),
            "shadow": False,
            "topLevel": False,
        }

    return hat_id


def _note_lane_specs(lane, ticks_per_beat):
    specs = []
    cursor = 0

    for note in lane:
        gap = note.start - cursor
        if gap > 0:
            specs.append({
                "opcode": "music_restForBeats",
                "inputs": {"BEATS": _number_input(gap / ticks_per_beat)},
            })

        # Instrument selection is repeated before each note so program
        # changes are reflected when a lane starts its next note.
        specs.append({
            "opcode": "music_setInstrument",
            "fields": {"INSTRUMENT": [str(note.instrument), None]},
        })
        specs.append({
            "opcode": "music_playNoteForBeats",
            "inputs": {
                "NOTE": _number_input(note.pitch),
                "BEATS": _number_input(
                    (note.end - note.start) / ticks_per_beat
                ),
            },
        })
        cursor = note.end

    return specs


def _tempo_specs(midi, ticks_per_beat):
    tempo_events = []
    for track_index, track in enumerate(midi.tracks):
        for event_index, event in enumerate(track.events):
            if _event_value(event, "type") == "tempo":
                tempo = int(_event_value(event, "tempo", 0))
                if tempo <= 0:
                    continue
                tempo_events.append(
                    (int(_event_value(event, "tick", 0)),
                     track_index, event_index, tempo)
                )

    # In format 1, tempo is a file-wide map. Stable order resolves same-tick
    # events deterministically if malformed files contain conflicting values.
    tempo_events.sort(key=lambda item: (item[0], item[1], item[2]))

    if not tempo_events:
        return [{"opcode": "music_setTempo",
                 "inputs": {"TEMPO": _number_input(120)}}]

    specs = []
    cursor = 0
    current_tempo = 120

    for tick, _, _, microseconds_per_beat in tempo_events:
        beats = (tick - cursor) / ticks_per_beat
        if beats > 0:
            specs.append({
                "opcode": "music_restForBeats",
                "inputs": {"BEATS": _number_input(beats)},
            })

        bpm = 60_000_000 / microseconds_per_beat
        specs.append({
            "opcode": "music_setTempo",
            "inputs": {"TEMPO": _number_input(round(bpm, 6))},
        })
        current_tempo = bpm
        cursor = tick

    return specs


def _svg_costume(name):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1" '
        'viewBox="0 0 1 1"><rect width="1" height="1" fill="#ffffff" '
        'fill-opacity="0"/></svg>'
    ).encode("utf-8")
    asset_id = hashlib.md5(svg).hexdigest()
    return {
        "assetId": asset_id,
        "name": name,
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "md5ext": f"{asset_id}.svg",
        " rotationCenterX": 0,
        " rotationCenterY": 0,
        "rotationCenterX": 0,
        "rotationCenterY": 0,
    }, svg


def _base_target(name, is_stage, layer_order):
    costume_name = "backdrop1" if is_stage else "costume1"
    costume, asset = _svg_costume(costume_name)

    target = {
        "isStage": is_stage,
        "name": name,
        "variables": {},
        "lists": {},
        "broadcasts": {},
        "blocks": {},
        "comments": {},
        "currentCostume": 0,
        "costumes": [costume],
        "sounds": [],
        "volume": 100,
        "layerOrder": layer_order,
    }

    if is_stage:
        target.update({
            "tempo": 120,
            "videoTransparency": 50,
            "videoState": "on",
            "textToSpeechLanguage": None,
        })
    else:
        target.update({
            "visible": True,
            "x": 0,
            "y": 0,
            "size": 100,
            "direction": 90,
            "draggable": False,
            "rotationStyle": "all around",
        })

    return target, {f"{costume['assetId']}.svg": asset}


def _build_stage(midi, counter, layer_order=0):
    stage, assets = _base_target("Stage", True, layer_order)
    tempo_specs = _tempo_specs(midi, midi.ticks_per_beat)
    _make_chain(stage["blocks"], counter, tempo_specs, 40, 40)
    return stage, assets


def _build_sprite(track, track_index, ticks_per_beat, counter, layer_order):
    name = track.name.strip() if track.name else f"Track {track_index + 1}"
    if not name:
        name = f"Track {track_index + 1}"

    sprite, assets = _base_target(name, False, layer_order)
    notes = _track_notes(track)
    lanes = _assign_lanes(notes)

    for lane_index, lane in enumerate(lanes):
        specs = _note_lane_specs(lane, ticks_per_beat)
        _make_chain(
            sprite["blocks"],
            counter,
            specs,
            40,
            40 + lane_index * 120,
        )

    return sprite, assets, len(notes), len(lanes)


def build_sb3(midi, output_path, project_name="MIDI Project"):
    """Build an SB3 file from a parsed MIDIFile object.

    Returns a summary dictionary. Returns None is not used here; parse errors
    and unsupported formats raise SB3BuildError.
    """
    if midi is None:
        raise SB3BuildError("MIDI parsing was skipped (polyphony limit reached)")

    if midi.format == 2:
        raise SB3BuildError(
            "MIDI Format 2 contains independent sequences; it is not "
            "supported by this initial SB3 builder."
        )

    if midi.ticks_per_beat <= 0:
        raise SB3BuildError("Invalid ticks_per_beat")

    counter = [0]
    stage, stage_assets = _build_stage(midi, counter)

    targets = [stage]
    assets = dict(stage_assets)
    summary = []

    for index, track in enumerate(midi.tracks):
        sprite, sprite_assets, note_count, lane_count = _build_sprite(
            track,
            index,
            midi.ticks_per_beat,
            counter,
            index + 1,
        )
        targets.append(sprite)
        assets.update(sprite_assets)
        summary.append({
            "track": index,
            "name": sprite["name"],
            "notes": note_count,
            "voices": lane_count,
        })

    project = {
        "targets": targets,
        "monitors": [],
        "extensions": ["music"],
        "meta": {
            "semver": "3.0.0",
            "vm": "0.2.0",
            "agent": "midi-to-sb3",
        },
    }

    output_path = os.fspath(output_path)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "project.json",
            json.dumps(project, ensure_ascii=False, separators=(",", ":")),
        )
        for asset_name, asset_data in assets.items():
            archive.writestr(asset_name, asset_data)

    return {
        "output": output_path,
        "tracks": len(midi.tracks),
        "notes": sum(item["notes"] for item in summary),
        "voices": sum(item["voices"] for item in summary),
        "track_details": summary,
    }


def build_from_file(midi_path, output_path, project_name="MIDI Project"):
    """Parse a MIDI file (including the polyphony pre-check) and build SB3."""
    from midi_parser import parse_midi

    midi = parse_midi(midi_path)
    if midi is None:
        raise SB3BuildError(
            "Polyphony limit reached; MIDI parser did not run."
        )
    return build_sb3(midi, output_path, project_name=project_name)
