from typing import List


def generate_tja(
    output: List[int],
    title="Untitled",
    subtitle="Generated by AI",
    wave="wave.ogg",
    bpm=240,
    offset=0,
    course="Hard",
    level=7,
    note_per_line=200,
) -> str:
    tja = f"TITLE: {title}\nSUBTITLE: {subtitle}\nWAVE: {wave}\nBPM: {bpm}\nOFFSET:{offset}\n\n"
    tja += f"COURSE:{course}\nLEVEL:{level}\n\n"

    tja += "#START\n"
    for i in range(0, len(output), note_per_line):
        chunk = output[i : i + note_per_line]
        chunk += [0] * (note_per_line - len(chunk))
        tja += "".join(map(str, chunk)) + ",\n"
    tja += "#END\n"

    return tja
