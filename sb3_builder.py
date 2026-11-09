"""Build data-driven Scratch 3.0 projects from MIDI files.

Architecture:
- MIDI note events are stored in Stage lists inside project.json.
- One hidden worker sprite creates one clone per note event.
- Each note clone plays one note and deletes itself.
- MIDI tempo changes are converted to absolute seconds. Scratch Music is
  held at 60 BPM so note durations and start delays use seconds directly.

The Scratch Music extension has a separate audio concurrency limit. The
builder reports when MIDI polyphony exceeds the extension's documented
30-sound threshold; clones do not bypass that audio limit.
"""

import bisect
import hashlib
import json
import os
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass


class SB3BuildError(Exception):
    """Raised when MIDI data cannot be represented as an SB3 project."""
    pass


SCRATCH_MAX_CLONES = 300
MUSIC_CONCURRENCY_LIMIT = 30
MUSIC_FIXED_TEMPO = 60.0
MAX_MUSIC_BEATS = 100.0


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
    """Pair Note On/Off events and retain the 1-based GM program number."""
    active = defaultdict(deque)
    programs = defaultdict(int)
    notes = []
    events = sorted(
        enumerate(track.events),
        key=lambda item: (int(_event_value(item[1], "tick", 0)), item[0])
    )
    last_tick = 0

    for _, event in events:
        tick = int(_event_value(event, "tick", 0))
        last_tick = max(last_tick, tick)
        event_type = _event_value(event, "type")
        channel = int(_event_value(event, "channel", 0) or 0)

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
                    notes.append(Note(
                        start=start,
                        end=tick,
                        pitch=max(0, min(130, pitch)),
                        velocity=velocity,
                        channel=channel,
                        instrument=program + 1,
                    ))

    for (channel, pitch), queue in active.items():
        while queue:
            start, velocity, program = queue.popleft()
            if last_tick > start:
                notes.append(Note(
                    start=start,
                    end=last_tick,
                    pitch=max(0, min(130, pitch)),
                    velocity=velocity,
                    channel=channel,
                    instrument=program + 1,
                ))

    return notes


def _build_tempo_converter(midi):
    """Return tick->seconds using the file-wide MIDI tempo map."""
    events = []
    for track_index, track in enumerate(midi.tracks):
        for event_index, event in enumerate(track.events):
            if _event_value(event, "type") != "tempo":
                continue
            tempo = int(_event_value(event, "tempo", 0) or 0)
            if tempo > 0:
                events.append((
                    int(_event_value(event, "tick", 0)),
                    track_index,
                    event_index,
                    tempo,
                ))

    events.sort(key=lambda item: (item[0], item[1], item[2]))

    # MIDI's default tempo is 500,000 microseconds per quarter note.
    tempo_points = [(0, 500000)]
    for tick, _, _, tempo in events:
        if tick == tempo_points[-1][0]:
            tempo_points[-1] = (tick, tempo)
        else:
            tempo_points.append((tick, tempo))

    ticks = []
    seconds_at_tick = []
    tempos = []
    elapsed = 0.0
    previous_tick = 0
    previous_tempo = tempo_points[0][1]

    for index, (tick, tempo) in enumerate(tempo_points):
        if index:
            elapsed += (
                (tick - previous_tick)
                * previous_tempo
                / 1_000_000
                / midi.ticks_per_beat
            )
        ticks.append(tick)
        seconds_at_tick.append(elapsed)
        tempos.append(tempo)
        previous_tick = tick
        previous_tempo = tempo

    def tick_to_seconds(tick):
        tick = max(0, int(tick))
        index = bisect.bisect_right(ticks, tick) - 1
        return (
            seconds_at_tick[index]
            + (tick - ticks[index])
            * tempos[index]
            / 1_000_000
            / midi.ticks_per_beat
        )

    return tick_to_seconds


def _max_polyphony(notes):
    """Return the exact maximum number of overlapping note intervals."""
    events = []
    for note in notes:
        events.append((note.start, 1))
        events.append((note.end, -1))

    # Note Offs are processed before Note Ons at the same tick.
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    maximum = 0
    for _, change in events:
        active += change
        maximum = max(maximum, active)
    return maximum


def _new_id(counter):
    counter[0] += 1
    return f"b{counter[0]:08x}"


def _number_input(value):
    return [1, [4, str(value)]]


def _reporter_input(block_id, shadow_type=4, default="0"):
    return [3, block_id, [shadow_type, str(default)]]


def _add_block(blocks, counter, opcode, parent=None, inputs=None,
               fields=None, top_level=False, x=0, y=0, shadow=False):
    block_id = _new_id(counter)
    block = {
        "opcode": opcode,
        "next": None,
        "parent": parent,
        "inputs": inputs or {},
        "fields": fields or {},
        "shadow": shadow,
        "topLevel": top_level,
    }
    if top_level:
        block["x"] = x
        block["y"] = y
    blocks[block_id] = block
    return block_id


def _add_variable_reporter(blocks, counter, variable_id, name, parent):
    return _add_block(
        blocks, counter, "data_variable", parent=parent,
        fields={"VARIABLE": [name, variable_id]}
    )


def _add_list_item_reporter(blocks, counter, list_id, list_name,
                            index_variable_id, index_name, parent):
    reporter_id = _add_block(
        blocks, counter, "data_itemoflist", parent=parent,
        fields={"LIST": [list_name, list_id]}
    )
    index_id = _add_variable_reporter(
        blocks, counter, index_variable_id, index_name, reporter_id
    )
    blocks[reporter_id]["inputs"]["INDEX"] = _reporter_input(
        index_id, shadow_type=6, default="1"
    )
    return reporter_id


def _add_list_length_reporter(blocks, counter, list_id, list_name, parent):
    return _add_block(
        blocks, counter, "data_lengthoflist", parent=parent,
        fields={"LIST": [list_name, list_id]}
    )


def _add_subtract_reporter(blocks, counter, left_id, left_name,
                           right_id, right_name, parent):
    reporter_id = _add_block(
        blocks, counter, "operator_subtract", parent=parent
    )
    left = _add_variable_reporter(
        blocks, counter, left_id, left_name, reporter_id
    )
    right = _add_variable_reporter(
        blocks, counter, right_id, right_name, reporter_id
    )
    blocks[reporter_id]["inputs"] = {
        "NUM1": _reporter_input(left, 4, "0"),
        "NUM2": _reporter_input(right, 4, "0"),
    }
    return reporter_id


def _link(blocks, previous_id, next_id):
    if previous_id is not None:
        blocks[previous_id]["next"] = next_id


def _build_worker_sprite(stage_lists, counter, layer_order=1):
    """Build a hidden manager sprite and a one-note-per-clone worker."""
    sprite_id = "worker-sprite"
    variable_names = [
        "index", "previousStart", "noteStart", "noteDuration",
        "notePitch", "noteInstrument",
    ]
    variables = {
        f"var_{name}": [name, 0, False]
        for name in variable_names
    }
    var_ids = {name: f"var_{name}" for name in variable_names}

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1" '
        'viewBox="0 0 1 1"><rect width="1" height="1" '
        'fill="#ffffff" fill-opacity="0"/></svg>'
    ).encode("utf-8")
    asset_id = hashlib.md5(svg).hexdigest()
    costume = {
        "assetId": asset_id,
        "name": "costume1",
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "md5ext": f"{asset_id}.svg",
        "rotationCenterX": 0,
        "rotationCenterY": 0,
    }

    sprite = {
        "isStage": False,
        "name": "Note Worker",
        "variables": variables,
        "lists": {},
        "broadcasts": {},
        "blocks": {},
        "comments": {},
        "currentCostume": 0,
        "costumes": [costume],
        "sounds": [],
        "volume": 100,
        "layerOrder": layer_order,
        "visible": False,
        "x": 0,
        "y": 0,
        "size": 1,
        "direction": 90,
        "draggable": False,
        "rotationStyle": "all around",
    }
    blocks = sprite["blocks"]

    # Manager: initialize the fixed tempo and spawn one clone for each note.
    flag = _add_block(
        blocks, counter, "event_whenflagclicked",
        top_level=True, x=40, y=40
    )
    set_tempo = _add_block(
        blocks, counter, "music_setTempo", parent=flag,
        inputs={"TEMPO": _number_input(MUSIC_FIXED_TEMPO)}
    )
    blocks[flag]["next"] = set_tempo

    hide = _add_block(blocks, counter, "looks_hide", parent=set_tempo)
    blocks[set_tempo]["next"] = hide

    set_index = _add_block(
        blocks, counter, "data_setvariableto", parent=hide,
        fields={"VARIABLE": ["index", var_ids["index"]]},
        inputs={"VALUE": _number_input(1)}
    )
    blocks[hide]["next"] = set_index

    set_previous = _add_block(
        blocks, counter, "data_setvariableto", parent=set_index,
        fields={"VARIABLE": ["previousStart", var_ids["previousStart"]]},
        inputs={"VALUE": _number_input(0)}
    )
    blocks[set_index]["next"] = set_previous

    repeat = _add_block(
        blocks, counter, "control_repeat", parent=set_previous
    )
    blocks[set_previous]["next"] = repeat
    length_reporter = _add_list_length_reporter(
        blocks, counter, stage_lists["StartSeconds"][0],
        "StartSeconds", repeat
    )
    blocks[repeat]["inputs"]["TIMES"] = _reporter_input(
        length_reporter, 6, "0"
    )

    # Body: set event data, wait until its start, then create its clone.
    body = []
    command_specs = [
        ("data_setvariableto", "noteStart"),
        ("control_wait", None),
        ("data_setvariableto", "noteDuration"),
        ("data_setvariableto", "notePitch"),
        ("data_setvariableto", "noteInstrument"),
        ("control_create_clone_of", None),
        ("data_setvariableto", "previousStart"),
        ("data_changevariableby", "index"),
    ]
    body_ids = []
    for opcode, _ in command_specs:
        body_ids.append(_add_block(
            blocks, counter, opcode, parent=repeat
        ))
    for left, right in zip(body_ids, body_ids[1:]):
        _link(blocks, left, right)
    blocks[repeat]["inputs"]["SUBSTACK"] = [2, body_ids[0]]

    # noteStart = item(index) of StartSeconds
    reporter = _add_list_item_reporter(
        blocks, counter, stage_lists["StartSeconds"][0], "StartSeconds",
        var_ids["index"], "index", body_ids[0]
    )
    blocks[body_ids[0]]["fields"] = {
        "VARIABLE": ["noteStart", var_ids["noteStart"]]
    }
    blocks[body_ids[0]]["inputs"]["VALUE"] = _reporter_input(reporter)

    # wait (noteStart - previousStart) seconds
    difference = _add_subtract_reporter(
        blocks, counter,
        var_ids["noteStart"], "noteStart",
        var_ids["previousStart"], "previousStart",
        body_ids[1]
    )
    blocks[body_ids[1]]["inputs"]["DURATION"] = _reporter_input(
        difference, 4, "0"
    )

    # Remaining event fields from Stage lists.
    for block_id, variable_name, list_name in [
        (body_ids[2], "noteDuration", "DurationSeconds"),
        (body_ids[3], "notePitch", "Pitch"),
        (body_ids[4], "noteInstrument", "Instrument"),
    ]:
        reporter = _add_list_item_reporter(
            blocks, counter, stage_lists[list_name][0], list_name,
            var_ids["index"], "index", block_id
        )
        blocks[block_id]["fields"] = {
            "VARIABLE": [variable_name, var_ids[variable_name]]
        }
        blocks[block_id]["inputs"]["VALUE"] = _reporter_input(reporter)

    # Create clone of this sprite. The menu is a shadow block.
    menu_id = _add_block(
        blocks, counter, "control_create_clone_of_menu",
        parent=body_ids[5],
        fields={"CLONE_OPTION": ["_myself_", None]},
        shadow=True
    )
    blocks[body_ids[5]]["inputs"]["CLONE_OPTION"] = [1, menu_id]

    # previousStart = noteStart
    reporter = _add_variable_reporter(
        blocks, counter, var_ids["noteStart"], "noteStart", body_ids[6]
    )
    blocks[body_ids[6]]["fields"] = {
        "VARIABLE": ["previousStart", var_ids["previousStart"]]
    }
    blocks[body_ids[6]]["inputs"]["VALUE"] = _reporter_input(reporter)

    blocks[body_ids[7]]["fields"] = {
        "VARIABLE": ["index", var_ids["index"]]
    }
    blocks[body_ids[7]]["inputs"]["VALUE"] = _number_input(1)

    # Clone worker: select this note's GM instrument, play it, then self-delete.
    clone_hat = _add_block(
        blocks, counter, "control_start_as_clone",
        top_level=True, x=420, y=40
    )
    set_instrument = _add_block(
        blocks, counter, "music_midiSetInstrument", parent=clone_hat
    )
    blocks[clone_hat]["next"] = set_instrument
    instrument_reporter = _add_variable_reporter(
        blocks, counter, var_ids["noteInstrument"],
        "noteInstrument", set_instrument
    )
    blocks[set_instrument]["inputs"]["INSTRUMENT"] = _reporter_input(
        instrument_reporter, 4, "1"
    )

    play_note = _add_block(
        blocks, counter, "music_playNoteForBeats", parent=set_instrument
    )
    blocks[set_instrument]["next"] = play_note
    pitch_reporter = _add_variable_reporter(
        blocks, counter, var_ids["notePitch"], "notePitch", play_note
    )
    duration_reporter = _add_variable_reporter(
        blocks, counter, var_ids["noteDuration"], "noteDuration", play_note
    )
    blocks[play_note]["inputs"] = {
        "NOTE": _reporter_input(pitch_reporter, 4, "60"),
        "BEATS": _reporter_input(duration_reporter, 4, "0.25"),
    }

    delete_clone = _add_block(
        blocks, counter, "control_delete_this_clone", parent=play_note
    )
    blocks[play_note]["next"] = delete_clone

    return sprite, {f"{asset_id}.svg": svg}


def _build_stage(note_rows):
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1" '
        'viewBox="0 0 1 1"><rect width="1" height="1" '
        'fill="#ffffff" fill-opacity="0"/></svg>'
    ).encode("utf-8")
    asset_id = hashlib.md5(svg).hexdigest()

    list_values = {
        "StartSeconds": [row[0] for row in note_rows],
        "DurationSeconds": [row[1] for row in note_rows],
        "Pitch": [row[2] for row in note_rows],
        "Instrument": [row[3] for row in note_rows],
    }
    lists = {}
    stage_lists = {}
    for name, values in list_values.items():
        list_id = f"list_{name}"
        lists[list_id] = [name, values, False]
        stage_lists[name] = (list_id, values)

    stage = {
        "isStage": True,
        "name": "Stage",
        "variables": {},
        "lists": lists,
        "broadcasts": {},
        "blocks": {},
        "comments": {},
        "currentCostume": 0,
        "costumes": [{
            "assetId": asset_id,
            "name": "backdrop1",
            "bitmapResolution": 1,
            "dataFormat": "svg",
            "md5ext": f"{asset_id}.svg",
            "rotationCenterX": 0,
            "rotationCenterY": 0,
        }],
        "sounds": [],
        "volume": 100,
        "layerOrder": 0,
        "tempo": MUSIC_FIXED_TEMPO,
        "videoTransparency": 50,
        "videoState": "on",
        "textToSpeechLanguage": None,
    }
    return stage, stage_lists, {f"{asset_id}.svg": svg}


def build_sb3(midi, output_path, project_name="MIDI Project"):
    """Build an SB3 project using Stage lists and one clone per note."""
    if midi is None:
        raise SB3BuildError("MIDI parsing was skipped (polyphony limit reached)")
    if midi.format == 2:
        raise SB3BuildError(
            "MIDI Format 2 contains independent sequences and is not supported."
        )
    if midi.ticks_per_beat <= 0:
        raise SB3BuildError("Invalid ticks_per_beat")

    notes = []
    for track in midi.tracks:
        notes.extend(_track_notes(track))
    notes.sort(key=lambda note: (note.start, note.end, note.channel, note.pitch))

    maximum = _max_polyphony(notes)
    if maximum >= SCRATCH_MAX_CLONES:
        raise SB3BuildError(
            f"Maximum polyphony is {maximum}; Scratch allows at most "
            f"{SCRATCH_MAX_CLONES} clones at once."
        )

    tick_to_seconds = _build_tempo_converter(midi)
    rows = []
    for note in notes:
        start_seconds = tick_to_seconds(note.start)
        end_seconds = tick_to_seconds(note.end)
        duration_seconds = max(0.001, end_seconds - start_seconds)
        rows.append((
            round(start_seconds, 6),
            round(duration_seconds, 6),
            int(note.pitch),
            int(note.instrument),
        ))
    rows.sort(key=lambda row: row[0])

    stage, stage_lists, stage_assets = _build_stage(rows)
    counter = [0]
    worker, worker_assets = _build_worker_sprite(stage_lists, counter)

    project = {
        "targets": [stage, worker],
        "monitors": [],
        "extensions": ["music"],
        "meta": {
            "semver": "3.0.0",
            "vm": "0.2.0",
            "agent": "midi-to-sb3 clone-list builder",
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
        for name, data in {**stage_assets, **worker_assets}.items():
            archive.writestr(name, data)

    return {
        "output": output_path,
        "tracks": len(midi.tracks),
        "notes": len(notes),
        "voices": maximum,
        "max_polyphony": maximum,
        "music_concurrency_limit": MUSIC_CONCURRENCY_LIMIT,
        "warnings": (
            [f"Polyphony {maximum} exceeds Scratch Music's "
             f"{MUSIC_CONCURRENCY_LIMIT}-sound concurrency threshold; "
             "some notes may be skipped."]
            if maximum > MUSIC_CONCURRENCY_LIMIT else []
        ),
        "list_items": len(rows),
    }


def build_from_file(midi_path, output_path, project_name="MIDI Project"):
    """Parse a MIDI file (including the clone-capacity pre-check) and build SB3."""
    from midi_parser import parse_midi

    midi = parse_midi(midi_path)
    if midi is None:
        raise SB3BuildError(
            "Polyphony limit reached; MIDI parser did not run."
        )
    return build_sb3(midi, output_path, project_name=project_name)


def main():
    import sys

    if len(sys.argv) < 2:
        filename = os.path.basename(__file__)
        print(f"Usage: python {filename} input.mid [output.sb3]")
        return 1

    input_path = sys.argv[1]
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as e:
            print(f"[ERROR] Tkinter is unavailable: {e}")
            return 1

        input_abs = os.path.abspath(input_path)
        root = tk.Tk()
        root.withdraw()
        try:
            output_path = filedialog.asksaveasfilename(
                parent=root,
                title="Save Scratch project",
                initialdir=os.path.dirname(input_abs),
                initialfile=os.path.splitext(os.path.basename(input_abs))[0] + ".sb3",
                defaultextension=".sb3",
                filetypes=[("Scratch 3 project", "*.sb3"), ("All files", "*.*")]
            )
        finally:
            root.destroy()

        if not output_path:
            print("[CANCEL] Save cancelled.")
            return 0

    try:
        result = build_from_file(input_path, output_path)
        print("[SUCCESS] SB3 generated")
        print(f"Output: {result['output']}")
        print(f"Tracks: {result['tracks']}")
        print(f"Notes: {result['notes']}")
        print(f"Max Polyphony / Clones: {result['max_polyphony']}")
        print(f"Stage list items: {result['list_items']}")
        for warning in result["warnings"]:
            print(f"[WARNING] {warning}")
        return 0
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
