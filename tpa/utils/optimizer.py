import math
from typing import Callable

import torch

__all__ = [
    "OptimizerFactory",
    "get_lr_scheduler",
    "warmup_poly_lr",
    "warmup_cosine_annealing_lr",
]


class OptimizerFactory:
    """Factory class for creating optimizers"""

    @staticmethod
    def get_optimizer(name: str, parameters, **kwargs) -> torch.optim.Optimizer:
        """
        Create optimizer instance based on name and parameters

        Args:
            name: Optimizer name ('sgd', 'adam', or 'adamw')
            parameters: Model parameters to optimize
            **kwargs: Optimizer-specific parameters
        """
        name = name.lower()
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=kwargs.get("lr", 0.01),
                momentum=kwargs.get("momentum", 0.9),
                weight_decay=kwargs.get("weight_decay", 1e-4),
                nesterov=kwargs.get("nesterov", False),
            )
        elif name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=kwargs.get("lr", 0.001),
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 1e-4),
                amsgrad=kwargs.get("amsgrad", False),
            )
        elif name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=kwargs.get("lr", 0.001),
                betas=kwargs.get("betas", (0.9, 0.999)),
                eps=kwargs.get("eps", 1e-8),
                weight_decay=kwargs.get("weight_decay", 0.01),
                amsgrad=kwargs.get("amsgrad", False),
            )
        else:
            raise ValueError(
                f"Unsupported optimizer: {name}. "
                "Choose from 'sgd', 'adam', or 'adamw'"
            )


def warmup_poly_lr(
    initial_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    power: float = 1.0,
) -> Callable[[int], float]:
    """
    Creates a learning rate scheduler that warms up from `initial_lr` to `min_lr` over `total_steps` steps.
    The learning rate is increased linearly from 0 to `initial_lr` over `warmup_steps` steps, and then
    decayed polynomially to `min_lr` over the remaining `total_steps - warmup_steps` steps.

    Args:
        initial_lr (float): The initial learning rate.
        min_lr (float): The minimum learning rate.
        warmup_steps (int): The number of steps to warm up the learning rate.
        total_steps (int): The total number of steps.
        power (float, optional): The power to use for the polynomial decay. Defaults to 1.0.

    Returns:
        A callable that takes an integer `step` and returns the learning rate at that step.
    """

    def scheduler(step: int) -> float:
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        else:
            decay = (1 - (step - warmup_steps) / (total_steps - warmup_steps)) ** power
            return max(min_lr, initial_lr * decay)

    return scheduler


def warmup_cosine_annealing_lr(
    initial_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> Callable[[int], float]:
    """
    Creates a learning rate scheduler that warms up from `initial_lr` to `min_lr` over `total_steps` steps.
    The learning rate is increased linearly from 0 to `initial_lr` over `warmup_steps` steps, and then
    decayed cosinically to `min_lr` over the remaining `total_steps - warmup_steps` steps.

    Args:
        initial_lr (float): The initial learning rate.
        min_lr (float): The minimum learning rate.
        warmup_steps (int): The number of steps to warm up the learning rate.
        total_steps (int): The total number of steps.

    Returns:
        A callable that takes an integer `step` and returns the learning rate at that step.
    """

    def scheduler(step: int) -> float:
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (progress % 1)))
            return min_lr + (initial_lr - min_lr) * cosine_decay

    return scheduler


def get_lr_scheduler(
    scheduler_type: str,
    total_steps: int,
    initial_lr: float,
    min_lr: float,
    warmup_steps: int,
    update_frequency: str = "step",
    **kwargs,
) -> Callable[[int], float]:
    """
    Creates a learning rate scheduler.

    Args:
        scheduler_type: The type of learning rate scheduler. Currently supports "poly" and "cosine".
        initial_lr: The initial learning rate.
        min_lr: The minimum learning rate.
        warmup_steps: The number of steps to warm up the learning rate.
        total_steps: The total number of steps.
        power: The power to use for the polynomial decay. Defaults to 1.0.
        update_frequency: The frequency at which the learning rate is updated. Defaults to "step".

    Returns:
        A callable that takes an integer `step` or `epoch` and returns the learning rate at that step or epoch.
    """
    if scheduler_type == "poly":
        base_scheduler = warmup_poly_lr(
            initial_lr,
            min_lr,
            warmup_steps,
            total_steps,
            kwargs.get("power", 0.9),
        )
    elif scheduler_type == "cosine":
        base_scheduler = warmup_cosine_annealing_lr(
            initial_lr,
            min_lr,
            warmup_steps,
            total_steps,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Choose from 'poly' or 'cosine'"
        )

    if update_frequency == "step":
        return base_scheduler
    elif update_frequency == "epoch":

        def epoch_scheduler(epoch: int) -> float:
            return base_scheduler(epoch * (total_steps // warmup_steps))

        return epoch_scheduler
    else:
        raise ValueError(
            f"Unknown update frequency: {update_frequency}. Choose from 'step' or 'epoch'"
        )
