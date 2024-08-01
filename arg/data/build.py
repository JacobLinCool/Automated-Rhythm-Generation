import os
import sys
import json
from glob import glob
from datasets import Dataset, Audio

data_dir = sys.argv[1]
name = sys.argv[2]
if not os.path.exists(data_dir):
    print(f"Data directory {data_dir} does not exist.")
    sys.exit(1)
if not name:
    print(f"Dataset repository {name} is not valid.")
    sys.exit(1)


def read_tja(tja_path):
    print(tja_path)
    try:
        with open(tja_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(tja_path, "r", encoding="shift-jis") as f:
            return f.read()


def read_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def read_notes(json_path):
    json = read_json(json_path)
    data = json["data"]

    notes = {}
    for d in data:
        # d is a dictionary
        if d["course"] == 0:
            notes["easy"] = d["chart"]
        elif d["course"] == 1:
            notes["normal"] = d["chart"]
        elif d["course"] == 2:
            notes["hard"] = d["chart"]
        elif d["course"] == 3:
            notes["oni"] = d["chart"]
        elif d["course"] == 4:
            notes["ura"] = d["chart"]

    return notes


musics = glob(data_dir + "/**/*.ogg", recursive=True)
print("Number of musics:", len(musics))

# get tuples of (music, tja)
examples = map(
    lambda music: (
        music,
        music.replace(".ogg", ".tja"),
        music.replace(".ogg", ".json"),
    ),
    musics,
)
examples = filter(
    lambda example: os.path.exists(example[1]) and os.path.exists(example[2]), examples
)
examples = map(
    lambda example: {
        "audio": example[0],
        "tja": read_tja(example[1]),
        **read_notes(example[2]),
    },
    examples,
)
examples = list(examples)
print("Number of examples:", len(examples))

# create dataset
dataset = Dataset.from_list(examples).cast_column("audio", Audio())
dataset.push_to_hub(name, private=True, token=os.getenv("HF_TOKEN"))
