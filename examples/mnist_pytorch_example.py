import copy
import platform
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

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

from epbt import Candidate, HyperParameters, epbt, hparam
from epbt.helpers import Config
import logging


@dataclass
class MnistConfig(Config):
    # input batch size for testing
    test_batch_size: int = 1000
    # For Saving the current Model
    save_model: bool = False


@dataclass
class HParams(HyperParameters):
    """ HyperParameters of the Mnist Classifier. """
    optimizer: str = "ADAM"
    # input batch size for training
    batch_size: int = hparam(64, min=1, max=1024)
    # number of epochs to train
    epochs: int = 1
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
    def __init__(self, hparams: HParams, config: MnistConfig):
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
            nn.Flatten(),  # type: ignore
            nn.Linear(self.hidden_size, 10),
        )
        self.loss = nn.CrossEntropyLoss()
        self.device = self.config.device
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.hparams.gamma)
        self.logger = self.config.get_logger(__file__)

    def forward(self, x):
        x_ = x.to(self.device)
        return self.classifier(self.encoder(x_)).to(x.device)

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.train()
        pbar = tqdm.tqdm(train_loader)
        pbar.set_description(f"Train Epoch: {epoch}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.shape[0]

            self.optimizer.zero_grad()
            output = self(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:

                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                accuracy = (correct / batch_size)
                self.logger.debug(f"accuracy: {accuracy:.2%}")
                pbar.set_postfix({"Loss": loss.item(), "accuracy": accuracy})

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
    
    def cuda(self, device: Union[int, torch.device, None]=None) -> "MnistClassifier":
        if isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device(device or "cuda")
        return super().cuda(self.device)
    
    def cpu(self):
        self.device = torch.device("cpu")
        super().cpu()


def accuracy(output: Tensor, target: Tensor) -> float:
    batch_size = output.shape[0]
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = (correct / batch_size)
    return accuracy


def parse_hparams_and_config() -> Tuple[HParams, MnistConfig]:
    # Training settings
    parser = ArgumentParser(description='PyTorch MNIST Example')
    parser.add_arguments(HParams, "hparams", default=HParams())
    parser.add_arguments(MnistConfig, "config", default=MnistConfig())
    args, _ = parser.parse_known_args()
    hparams: HParams = args.hparams
    config: MnistConfig = args.config
    return hparams, config

@dataclass
class MnistCandidate(Candidate):
    model: MnistClassifier = field(repr=False)
    hparams: HParams
    fitness: float


def get_dataloaders(batch_size: int, config: MnistConfig) -> Tuple[DataLoader, DataLoader]:
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
        batch_size=batch_size,
        shuffle=True, **config.dataloader_kwargs,
    )    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=True,
        **config.dataloader_kwargs,
    )
    return train_loader, test_loader


class DataParallelPassthrough(torch.nn.DataParallel):
    """ nn.DataParallel that allows access to the wrapped module attributes.

    Taken from https://github.com/pytorch/pytorch/issues/16885
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def evaluate_mnist_classifier(candidate: MnistCandidate=None) -> MnistCandidate:
    """Evaluation function, as described in `dummy_example.py` and `epbt.py`.
    
    Args:
        candidate (MnistCandidate): A previous candidate to start off of. If
        `None`, start from scratch. 
    
    Returns:
        MnistCandidate: the next candidate.
    """
    if candidate:
        config = candidate.model.config
        # get the (potentially mutated) hparams:
        hparams = candidate.hparams
    else:
        # if we don't have a previous 'Candiate' (first generation),
        # then parse the params and args from the command-line.
        hparams, config = parse_hparams_and_config()

    logger = config.get_logger(__file__)
    logger.info("Evaluating candidate with hparams: ", hparams)
    
    train_loader, test_loader = get_dataloaders(hparams.batch_size, config)

    # TODO: on Windows, can't use torch multiprocessing with CUDA tensors,
    # so learning is constrained to cpu-only. (see )
    if config.device.type == "cuda" and platform.system() == "Windows":
        config.device = torch.device("cpu")
        logger.warning("TODO: Cannot use torch.multiprocessing and cuda tensors on Windows for now.")
        # TODO: Try to see if we can get everything on the CPU and get it
        # to work. (I previously tried model.cpu(), and it wasn't enough...)
    
    # Create the model instance
    model = MnistClassifier(hparams, config)
    
    # Distribute the model to all gpus to reduce the load.
    if model.device.type == "cuda":
        model = DataParallelPassthrough(model)

    if candidate:
        old_model = candidate.model
        model.load_state_dict(old_model.state_dict())

    new_candidate = MnistCandidate(model, hparams, 0.)
    
    for epoch in range(1, hparams.epochs + 1):
        model.train_epoch(train_loader, epoch)
        fitness = model.test_epoch(test_loader)
        new_candidate.fitness = fitness

    if config.save_model:
        torch.save(model.state_dict(), config.log_dir / "mnist_cnn.pt")
    
    return new_candidate


@dataclass
class MnistExampleOptions:
    # Size of the population.
    population_size: int = 5
    # Number of generations (steps of the algorithm).
    n_generations: int = 3
    # Maximum number of processes to use in multiprocessing.
    n_processes: Optional[int] = 4


def main(options: MnistExampleOptions):
    # Create the first candidate from scratch.
    original = evaluate_mnist_classifier(None)
    initial_population: List[Candidate] = []
    # Here we fill the initial population with copies of the original.
    # NOTE: could also just as easily create/load many different candidates.
    initial_population.append(original)
    initial_population.extend(
        copy.deepcopy(original) for _ in range(options.population_size-1)
    )

    # invoke the epbt function to get a generator of best candidates.
    best_candidate_gen = epbt(
        n_generations=options.population_size,
        initial_population=initial_population,
        evaluation_function=evaluate_mnist_classifier,
        n_processes=options.n_processes,
        multiprocessing=torch.multiprocessing,
    )
    
    for i, best_candidate in enumerate(best_candidate_gen):
        print(f"Best candidate at step {i}: {best_candidate}")

if __name__ == '__main__':
    from epbt.utils import requires_import
    options = MnistExampleOptions()
    
    with requires_import("simple_parsing") as simple_parsing:
        from simple_parsing import ArgumentParser
        parser = ArgumentParser()
        parser.add_arguments(MnistExampleOptions, dest="options", default=MnistExampleOptions())
        args, unknown = parser.parse_known_args()
        options = args.options

    main(options)
