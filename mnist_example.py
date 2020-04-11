from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from simple_parsing import ArgumentParser
from torch import Tensor
from torch import nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from epbt import epbt
from hyperparameters import HyperParameters, hparam
from model import Candidate, Population


@dataclass
class Config:
    # input batch size for testing
    test_batch_size: int = 1000
    # disables CUDA training
    no_cuda: bool = False
    # random seed
    seed: int = 1
    # how many batches to wait before logging training status
    log_interval: int = 10
    # For Saving the current Model
    save_model: bool = False

    def __post_init__(self):
        use_cuda = not self.no_cuda and torch.cuda.is_available()
        torch.manual_seed(self.seed)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.dataloader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


@dataclass
class HParams(HyperParameters):
    """ HyperParameters of the Mnist Classifier. """
    # Which device to use.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer: str = "ADAM"
    # input batch size for training
    batch_size: int = hparam(64, min=1, max=1024)
    # number of epochs to train
    epochs: int = hparam(14, min=1, max=20)
    # learning rate
    learning_rate: float = hparam(1e-3, min=1e-6, max=1.0)
    # Learning rate step gamma
    gamma: float = hparam(0.7, min=0.1, max=0.99)



class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int=3,
                 padding: int=1,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return self.pool(x)


class MnistClassifier(nn.Module):
    def __init__(self, hparams: HParams, config: Config):
        self.hparams = hparams
        self.config = config
        self.hidden_size = 100
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, padding=1),
            ConvBlock(16, 32, kernel_size=3, padding=1),
            ConvBlock(32, self.hidden_size, kernel_size=3, padding=1),
            ConvBlock(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden_size, 10),
        )
        self.loss = nn.CrossEntropyLoss()
        self.device = self.hparams.device
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.hparams.gamma)
    
    def forward(self, x):
        return self.classifier(self.encoder(x))

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.train()
        pbar = tqdm.tqdm(train_loader)
        pbar.set_description(f"Train Epoch: {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({"Loss": loss.item()})
        self.scheduler.step(epoch=epoch)

    @torch.no_grad()
    def test_epoch(self, test_loader: DataLoader) -> float:
        self.eval()
        test_loss = 0.
        correct = 0.
        pbar = tqdm.tqdm(test_loader)
        pbar.set_description(f"Test")
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            output = self(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        
            accuracy = (correct / len(test_loader.dataset))
            pbar.set_postfix({"Accuracy": accuracy})

        test_loss /= len(test_loader.dataset)
        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2%})\n")
        return accuracy


def parse_hparams_and_config() -> Tuple[HParams, Config]:
    # Training settings
    parser = ArgumentParser(description='PyTorch MNIST Example')
    parser.add_arguments(HParams, "hparams", default=HParams())
    parser.add_arguments(Config, "config", default=Config())
    args = parser.parse_args()
    
    hparams: HParams = args.hparams
    config: Config = args.config
    return hparams, config

@dataclass
class MnistCandidate(Candidate):
    model: MnistClassifier
    hparams: HParams
    fitness: float


def evaluate_mnist_classifier(candidate: MnistCandidate) -> MnistCandidate:
    """Evaluation function, as described in `dummy_example.py` and `epbt.py`.
    
    Args:
        candidate (MnistCandidate): A previous candidate.
    
    Returns:
        MnistCandidate: the next candidate.
    """
    if candidate:
        config = candidate.model.config
        # get the (potentially mutated) hparams:
        hparams = candidate.hparams
    else:
        hparams, config = parse_hparams_and_config()

    train_dataset =  datasets.MNIST(
        'data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    )
    test_dataset = datasets.MNIST(
        'data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hparams.batch_size,
        shuffle=True, **config.dataloader_kwargs
    )    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=True,
        **config.dataloader_kwargs
    )
    
    model = MnistClassifier(hparams, config)
    
    if candidate:
        old_model = candidate.model
        model.load_state_dict(old_model.state_dict())

    new_candidate = MnistCandidate(model, hparams, 0.)

    for epoch in range(1, hparams.epochs + 1):
        model.train_epoch(train_loader, epoch)
        fitness = model.test_epoch(test_loader)
        new_candidate.fitness = fitness

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    
    return new_candidate


if __name__ == '__main__':
    initial_population = Population(evaluate_mnist_classifier(None) for i in range(2))
    pop_gen = epbt(3, initial_population, evaluate_mnist_classifier, n_processes=2)
    
    for i, pop in enumerate(pop_gen):
        print(f"Population hparams at step {i}: ", [c.hparams for c in pop])
    
    # candidate = evaluate_mnist_classifier(None)
    # candidate = evaluate_mnist_classifier(candidate)
