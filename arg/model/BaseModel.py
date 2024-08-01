import time
from typing import List
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class BaseModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()

    def postprocess(self, output: List[int], gap=0.04):
        gap = int(gap * 200)

        erase = 0
        for i in range(1, len(output)):
            if output[i] != 0:
                if erase > 0:
                    output[i] = 0
                else:
                    erase = gap
            erase -= 1

    def predict(self, audio_16k: np.ndarray, device=None, gap=0.04) -> List[int]:
        """
        Predict the notes from the input audio. (16kHz, 1 channel)
        """
        input = torch.from_numpy(audio_16k).to(torch.float32)
        input = input.unsqueeze(0).unsqueeze(-1)

        if device is None:
            device = next(self.parameters()).device

        input = input.to(device)

        with torch.no_grad():
            output = self.forward(input)
            output = output.argmax(dim=-1).squeeze().tolist()

        self.postprocess(output, gap=gap)
        return output

    def print_summary(self):
        print(self)
        print("Parameters:")
        total = sum(p.numel() for p in self.parameters())
        print(f"\tTotal     : {total:,}")
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\tTrainable : {trainable:,}")

    def dummy_test(self, length=10.0):
        print(f"Test with {length} seconds of dummy input")
        dummy_input = np.random.randn(int(16000 * length))
        self.eval()
        t = time.time()
        output = self.predict(dummy_input)
        print("Output Length:", len(output))
        print(f"Elapsed time: {time.time() - t:.2f} seconds")
