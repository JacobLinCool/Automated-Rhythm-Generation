# Automated Rhythm Generation

ARG: Automated Rhythm Generation.

Let's generate rhythm game maps automatically!

## Features

### Neural Networks

- RhythmGen: Generate rhythm sequence from music piece.
  - [x] RGGRU: RhythmGen with CNN feature extractor and GRU. (`460K`)
  - [x] RGGRUAT: RhythmGen with CNN feature extractor, GRU, and multi-head attention. (`720K`)
  - [x] RGTR: RhythmGen with CNN feature extractor and Transformer. (`15M`)
  - [x] RGRoFormer: RhythmGen with CNN feature extractor, Transformer and RoPE. (`420K`)
  - [ ] Others: RhythmGen with other architectures.
- RhythmRec: Reconstruct music piece from rhythm sequence.
  - [ ] No, we didn't implement this yet.

### Utils

- RhythmAnnotation (`ryan`): Rhythm annotation tool and format. See [ryan](https://github.com/JacobLinCool/rhythm-rs/blob/main/tja/examples/ryan.rs) for more details.

## Installation

```sh
pip install automated-rhythm-generation
```

## Training

```py
from arg import Trainer

trainer = Trainer(
    "RGGRU", # Model architecture
    "JacobLinCool/taiko-2023-1.1", # Dataset
    difficulty="hard",
    num_epochs=300,
    learning_rate=0.001,
    batch_size=32,
    max_length=10.0,
)

trainer.train(
    push="JacobLinCool/RhythmGenGRU-1-hard",
    hf_token=HF_TOKEN,
)
```

Or you can use the command line interface:

```sh
python -m arg.train RGGRU JacobLinCool/taiko-2023-1.1 --difficulty hard --push JacobLinCool/RhythmGenGRU-1-hard
```

## Inference

```py
from arg import RGGRU, generate_tja
import librosa

model = RGGRU.from_pretrained("JacobLinCool/RhythmGenGRU-1-hard")

audio, sr = librosa.load("path/to/music.mp3", sr=16000)
seq = model.predict(audio)
tja = generate_tja(seq)

with open("path/to/output.tja", "w") as f:
    f.write(tja)
```

Or you can use the command line interface:

```sh
python -m arg.infer RGGRU JacobLinCool/RhythmGenGRU-1-hard path/to/music.mp3
```

## Others

To see the model architecture: `python -m arg.model.RGGRU`, `python -m arg.model.RGRoFormer`, etc.
