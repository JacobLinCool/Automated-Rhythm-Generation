import argparse
import os
import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset, Audio
from accelerate import Accelerator
from .data import extract_features, split_examples
from .model import BaseModel, RGGRU, RGGRUAT, RGTransformer, RGRoFormer


def type_loss(preds: torch.Tensor, targets: torch.Tensor):
    # since 0 is more than 90% of the data, we reduce its weight
    # maybe we can try focal loss here
    weights = torch.tensor([0.1, 0.45, 0.45]).to(preds.device)
    return F.cross_entropy(preds.transpose(1, 2), targets, weight=weights)


def train(
    model: BaseModel,
    train_loader: DataLoader,
    accelerator: Accelerator | None = None,
    writer: SummaryWriter | None = None,
    num_epochs=10,
    learning_rate=0.001,
):
    if accelerator is None:
        accelerator = Accelerator()

    # Initialize model, optimizer, and loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    device = accelerator.device

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        # Initialize the progress bar
        with tqdm.tqdm(
            total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ) as pbar:
            for batch in train_loader:
                samples = batch["wave"].to(device)
                gt = batch["note"].to(device)
                samples = samples.unsqueeze(-1)

                # Forward pass
                pred = model(samples)

                # Compute losses
                loss = type_loss(pred, gt)

                # Backward pass and optimization
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()

                # Update running loss totals
                total_loss += loss.item()

                pbar.update(1)

                # Log metrics to TensorBoard
                if writer is not None:
                    global_step = epoch * num_batches + pbar.n
                    writer.add_scalar("Loss", loss.item(), global_step)

                # Update the progress bar
                pbar.set_postfix(
                    {
                        "Loss": total_loss / pbar.n,
                    }
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Model type")
    parser.add_argument("dataset", type=str, help="Dataset name")
    parser.add_argument(
        "--difficulty",
        type=str,
        default="hard",
        help="Difficulty level of the dataset (easy, medium, hard)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=10.0,
        help="Maximum length of the audio samples",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training with fp16",
    )
    parser.add_argument(
        "--push",
        type=str,
        default="",
        help="Push the model to the hub repository",
    )
    return parser.parse_args()


class Trainer:
    def __init__(
        self,
        type: str,
        dataset: str,
        difficulty="hard",
        num_epochs=300,
        learning_rate=0.001,
        batch_size=8,
        max_length=10.0,
        accelerator: Accelerator | None = None,
    ):
        self.type = type.lower()
        self.dataset = dataset
        self.difficulty = difficulty
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.accelerator = accelerator or Accelerator()

    def train(self, push: str | None = None, hf_token: str | None = None):
        # Load dataset
        dataset = load_dataset(self.dataset, token=hf_token)
        dataset = dataset["train"]
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        dataset = dataset.map(
            split_examples,
            fn_kwargs={"difficulty": self.difficulty, "max_length": self.max_length},
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=self.batch_size,
            num_proc=4,
        )
        dataset = dataset.map(
            extract_features,
            fn_kwargs={"max_length": self.max_length},
            remove_columns=dataset.column_names,
            batched=True,
            batch_size=self.batch_size,
            num_proc=4,
        )

        train_loader = DataLoader(
            dataset.with_format("torch"),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if self.type == "rggru":
            model = RGGRU()
        elif self.type == "rggruat":
            model = RGGRUAT()
        elif self.type == "rgtr":
            model = RGTransformer()
        elif self.type == "rgroformer":
            model = RGRoFormer()
        else:
            raise ValueError(f"Unknown model: {self.type}")

        writer = SummaryWriter()
        train(
            model,
            train_loader,
            accelerator=self.accelerator,
            writer=writer,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
        )

        if push:
            model.push_to_hub(push, token=hf_token)


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    dataset = args.dataset
    difficulty = args.difficulty
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    max_length = args.max_length
    use_fp16 = args.fp16
    push = args.push

    accelerator = Accelerator(mixed_precision="bf16" if use_fp16 else None)
    device = accelerator.device
    print(f"Device: {device}")

    trainer = Trainer(
        model_name,
        dataset,
        difficulty=difficulty,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        accelerator=accelerator,
    )

    trainer.train(
        push=push,
        hf_token=os.getenv("HF_TOKEN"),
    )
