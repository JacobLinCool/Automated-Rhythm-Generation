import numpy as np


def split_examples(examples, difficulty: str, max_length: float):
    new_examples = {"audio": [], "note": []}

    for i in range(len(examples["audio"])):
        audio = examples["audio"][i]
        labels = examples[difficulty][i]

        audio_array = audio["array"]
        sample_rate = audio["sampling_rate"]
        max_audio_length = int(max_length * sample_rate)
        audio_length = len(audio_array)

        num_chunks = int(np.ceil(audio_length / max_audio_length))

        for j in range(num_chunks):
            start = j * max_audio_length
            end = min(start + max_audio_length, audio_length)
            new_audio = audio_array[start:end]

            new_labels = []
            for onset_type, onset_time, _, _ in labels:
                if onset_type > 4:
                    continue
                onset_sample = int(onset_time * sample_rate)
                if start <= onset_sample < end:
                    if onset_type == 1 or onset_type == 3:
                        onset_type = 1
                    if onset_type == 2 or onset_type == 4:
                        onset_type = 2
                    new_onset_time = (onset_sample - start) / sample_rate
                    new_labels.append([onset_type, new_onset_time])

            if new_audio.shape[0] < max_audio_length:
                pad_length = max_audio_length - new_audio.shape[0]
                new_audio = np.pad(new_audio, (0, pad_length))

            new_examples["audio"].append(
                {"array": new_audio, "sampling_rate": sample_rate}
            )
            new_examples["note"].append(new_labels)

    return new_examples


def extract_features(examples, max_length: float):
    new_examples = {"wave": [], "note": []}

    for i in range(len(examples["audio"])):
        audio = examples["audio"][i]
        labels = examples["note"][i]

        samples = audio["array"]

        # pad samples to 80 samples
        num_samples = samples.shape[0]
        num_pad = 0
        if num_samples % 80 != 0:
            num_pad = 80 - num_samples % 80
            samples = np.pad(samples, (0, num_pad))

        num_frames = samples.shape[0] // 80
        note = np.zeros(num_frames, dtype=np.int8)

        for type, time in labels:
            frame_idx = min(int(time * num_frames / max_length), num_frames - 1)
            note[frame_idx] = type

        new_examples["wave"].append(samples)
        new_examples["note"].append(note)

    return new_examples
