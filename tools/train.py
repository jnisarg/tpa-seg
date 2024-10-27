import argparse
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.cuda.amp as amp
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.table import Table
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from tpa.datasets import DataloaderConfig, build_dataloaders
from tpa.models import build_network
from tpa.utils.metrics import MetricsCalculator
from tpa.utils.optimizer import OptimizerFactory, get_lr_scheduler
from tpa.utils.summary import log_model_stats
from tpa.utils.tpa_criterion import TPACriterion

# Configure warnings
warnings.filterwarnings(action="ignore")


class TrainingConfig:
    """Configuration class for training parameters"""

    def __init__(self, args: argparse.Namespace):
        self.exp_name = args.exp_name
        self.base_dir = args.base_dir
        self.rich_logging = args.rich_logging
        self.reset_exp = args.reset_exp
        self.device = args.device
        self.epochs = args.epochs
        self.save_freq = args.save_freq
        self.keep_last_n = args.keep_last_n
        self.log_interval = args.log_interval
        self.lr_update_freq = args.update_frequency
        self.seed = args.seed

        self.class_names = args.class_names

        # Optimizer configuration
        self.optimizer_config = {
            "type": args.optimizer,
            "params": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "momentum": getattr(args, "momentum", 0.9),
                "nesterov": getattr(args, "nesterov", False),
                "betas": (getattr(args, "beta1", 0.9), getattr(args, "beta2", 0.999)),
                "amsgrad": getattr(args, "amsgrad", False),
            },
        }

        # Scheduler configuration
        self.scheduler_config = {
            "type": args.lr_scheduler,
            "params": {
                "min_lr": args.min_lr,
                "warmup_steps": args.warmup_steps,
                "update_frequency": args.update_frequency,
                "poly_power": getattr(args, "poly_power", 0.9),
            },
        }


class ExperimentManager:
    """Manages experiment directories and logging setup"""

    def __init__(
        self, exp_name, base_dir="experiments", rich_logging=False, reset=False
    ):
        self.exp_name = exp_name
        self.base_dir = base_dir
        self.rich_logging = rich_logging
        self.exp_dir = os.path.join(base_dir, exp_name)

        # Create experiment directory structure
        self.dirs = {
            "exp": self.exp_dir,
            "checkpoints": os.path.join(self.exp_dir, "checkpoints"),
            "best_checkpoints": os.path.join(self.exp_dir, "checkpoints", "best"),
            "logs": os.path.join(self.exp_dir, "logs"),
            "tensorboard": os.path.join(self.exp_dir, "tensorboard"),
            "configs": os.path.join(self.exp_dir, "configs"),
        }

        self.console = Console()

        self._setup_directories(reset)
        self.logger = self._setup_logging()
        self.writer = self._setup_tensorboard()

        # Log experiment initialization
        self.logger.info(f"Initializing experiment: {exp_name}")
        self.logger.info(f"Experiment directory: {self.exp_dir}")
        self._log_directory_structure()

    def _setup_directories(self, reset):
        """Setup or reset experiment directories"""
        # Check if experiment directory exists
        if os.path.exists(self.exp_dir):
            if reset:
                self.logger = self._setup_temp_logging()
                self.logger.warning(
                    f"Resetting existing experiment directory: {self.exp_dir}\n"
                )
                shutil.rmtree(self.exp_dir)
            else:
                raise ValueError(
                    f"Experiment {self.exp_name} already exists. "
                    "Use reset=True to override."
                )

        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def _setup_temp_logging(self):
        """Setup temporary logger for initialization messages"""
        logger = logging.getLogger("temp")
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        return logger

    def _setup_logging(self):
        """Setup logging configuration"""
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(logging.INFO)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        simple_formatter = logging.Formatter("%(message)s")

        # Setup file handler
        log_file = os.path.join(
            self.dirs["logs"],
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)

        # Setup console handler
        if self.rich_logging:
            console_handler = RichHandler(console=self.console, rich_tracebacks=True)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    @staticmethod
    def clear_console():
        """Clear console"""
        os.system("cls" if os.name == "nt" else "clear")

    def _setup_tensorboard(self):
        """Setup TensorBoard writer"""
        return SummaryWriter(self.dirs["tensorboard"])

    def _log_directory_structure(self):
        """Log the directory structure of the experiment"""
        self.logger.info("Experiment directory structure:")
        for name, path in self.dirs.items():
            self.logger.info(f"{name}: {path}")

    def save_config(self, config_dict):
        """Save experiment configuration"""
        import json

        config_path = os.path.join(
            self.dirs["configs"],
            f'config_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
        )
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        self.logger.info(f"Configuration saved to {config_path}\n")


class MetricsTracker:
    """Tracks and logs training metrics"""

    def __init__(self, writer: SummaryWriter, logger: logging.Logger):
        self.writer = writer
        self.logger = logger
        self.best_loss = float("inf")
        self.best_iou = 0.0
        self.current_epoch = 0

    def update(
        self, epoch: int, train_loss: float, val_loss: float, val_iou: float
    ) -> bool:
        """
        Updates metrics and returns True if metrics improved
        """
        self.current_epoch = epoch
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("IoU/Validation", val_iou, epoch)

        improved = val_loss < self.best_loss or val_iou > self.best_iou
        if improved:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_iou = max(val_iou, self.best_iou)
            self.logger.info("\n🎯 New best metrics achieved!")
            self.log_best_metrics()

        return improved

    def log_best_metrics(self):
        """Logs the current best metrics"""
        self.logger.info(
            f"Best metrics:\n"
            f"Best Loss: {self.best_loss:.4f}\n"
            f"Best IoU: {self.best_iou:.4f}"
        )


class CheckpointManager:
    """Manages model checkpoints"""

    def __init__(self, checkpoint_dir: str, best_checkpoint_dir: str, keep_last_n: int):
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint_dir = best_checkpoint_dir
        self.keep_last_n = keep_last_n

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: amp.GradScaler,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
        logger: logging.Logger,
    ):
        """Saves model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            **metrics,
        }

        if is_best:
            path = os.path.join(
                self.best_checkpoint_dir,
                f"best_checkpoint_e{epoch}_loss_{metrics['mean_loss']:.4f}_iou_{metrics['mean_iou']:.4f}.pth",
            )
            torch.save(checkpoint, path)
            logger.info(f"\n📦 Saved best checkpoint: {path}")
        else:
            path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, path)
            logger.info(f"\n💾 Saved regular checkpoint: {path}")

        self._cleanup_old_checkpoints(logger)

    def _cleanup_old_checkpoints(self, logger: logging.Logger):
        """Removes old checkpoints keeping only the last N"""
        if self.keep_last_n <= 0:
            return

        for checkpoint_dir in [self.checkpoint_dir, self.best_checkpoint_dir]:
            checkpoints = sorted(
                Path(checkpoint_dir).glob("*.pth"), key=os.path.getmtime
            )
            if len(checkpoints) > self.keep_last_n:
                for checkpoint in checkpoints[: -self.keep_last_n]:
                    checkpoint.unlink()
                    logger.info(f"🗑️ Removed old checkpoint: {checkpoint}")


class Trainer:
    """Main training class"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.exp_manager = ExperimentManager(
            config.exp_name, config.base_dir, config.rich_logging, config.reset_exp
        )
        self.metrics_tracker = MetricsTracker(
            self.exp_manager.writer, self.exp_manager.logger
        )
        self.checkpoint_manager = CheckpointManager(
            self.exp_manager.dirs["checkpoints"],
            self.exp_manager.dirs["best_checkpoints"],
            config.keep_last_n,
        )
        self.logger = self.exp_manager.logger
        self.device = torch.device(config.device)
        self.console = self.exp_manager.console
        self._set_seed()
        # self.exp_manager.save_config(self.config)

    def _set_seed(self):
        """Sets random seeds for reproducibility"""
        torch.manual_seed(self.config.seed)
        if self.config.device == "cuda":
            torch.cuda.manual_seed(self.config.seed)

    def train(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
    ):
        """Main training loop"""
        model = model.to(self.device)
        optimizer = self._setup_optimizer(model)
        lr_scheduler = self._setup_scheduler(len(train_loader))
        scaler = amp.GradScaler()

        self._log_training_setup(model, optimizer, loss_fn)

        for epoch in range(self.config.epochs):
            self.logger.info(f"\n📈 Epoch {epoch + 1}/{self.config.epochs}")

            # Training phase
            train_loss = self._train_epoch(
                model, train_loader, optimizer, lr_scheduler, scaler, loss_fn, epoch
            )

            # Validation phase
            val_metrics = self._validate(model, val_loader, loss_fn)

            # Update metrics and save checkpoints
            improved = self.metrics_tracker.update(
                epoch + 1, train_loss, val_metrics["mean_loss"], val_metrics["mean_iou"]
            )

            self.logger.info(
                f"\nTraining Loss: {train_loss:.4f}\nValidation Loss: {val_metrics['mean_loss']:.4f}"
            )

            self.checkpoint_manager.save_checkpoint(
                model, optimizer, scaler, epoch + 1, val_metrics, improved, self.logger
            )

            self.console.print(Rule(f"Epoch {epoch + 1} complete"))

        self.logger.info("🎉 Training completed successfully!")
        self.exp_manager.writer.close()

    def _train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        scaler: amp.GradScaler,
        loss_fn: torch.nn.Module,
        epoch: int,
    ) -> float:
        """Trains the model for one epoch"""
        model.train()
        total_loss = 0
        steps_per_epoch = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch + 1}",
            leave=False,
            ncols=100,
            colour="blue",
        )

        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images, masks = images.to(self.device), masks.to(self.device)

            loss = self._train_step(model, images, masks, optimizer, scaler, loss_fn)
            total_loss += loss

            # Update learning rate if needed
            if self.config.lr_update_freq == "step":
                self._update_learning_rate(
                    optimizer, lr_scheduler, batch_idx + (epoch * steps_per_epoch)
                )

            # Update progress bar
            cur_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{cur_lr:.2e}"})

        if self.config.lr_update_freq == "epoch":
            self._update_learning_rate(optimizer, lr_scheduler, epoch)

        return total_loss / len(train_loader)

    def _train_step(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        masks: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: amp.GradScaler,
        loss_fn: torch.nn.Module,
    ) -> float:
        """Performs a single training step"""
        optimizer.zero_grad()

        with amp.autocast():
            outputs = model(images)
            loss = loss_fn(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return loss.item()

    @torch.no_grad()
    def _validate(
        self,
        model: torch.nn.Module,
        val_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
    ) -> Tuple[float, float]:
        """Validates the model"""
        model.eval()
        metrics_calculator = MetricsCalculator(
            n_classes=len(self.config.class_names),
            class_names=self.config.class_names,
            device=self.device,
        )

        progress_bar = tqdm(
            val_loader, desc="Validation", leave=False, ncols=100, colour="green"
        )

        with amp.autocast():
            for images, masks, _ in progress_bar:
                images, masks = images.to(self.device), masks.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = loss_fn(outputs, masks)

                # Convert outputs to predictions
                pred_masks = torch.argmax(outputs, dim=1)

                # Update metrics
                metrics_calculator.update(pred_masks, masks, loss.item())

        # Compute final metrics
        metrics = metrics_calculator.compute_metrics()
        metrics_calculator.log_metrics(metrics, self.console)

        return metrics

    def _setup_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Creates the optimizer"""
        return OptimizerFactory.get_optimizer(
            self.config.optimizer_config["type"],
            model.parameters(),
            **self.config.optimizer_config["params"],
        )

    def _setup_scheduler(self, steps_per_epoch: int) -> Any:
        """Creates the learning rate scheduler"""
        scheduler_params = {
            "total_steps": self.config.epochs * steps_per_epoch,
            "initial_lr": self.config.optimizer_config["params"]["lr"],
            **self.config.scheduler_config["params"],
        }
        return get_lr_scheduler(
            scheduler_type=self.config.scheduler_config["type"], **scheduler_params
        )

    def _update_learning_rate(
        self, optimizer: torch.optim.Optimizer, lr_scheduler: Any, step: int
    ):
        """Updates the learning rate"""
        new_lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def _log_training_setup(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ):
        """Logs training setup information"""
        self.logger.info("\n🚀 Training Setup:")
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Optimizer: {optimizer.__class__.__name__}")
        self.logger.info(f"Loss Function: {loss_fn.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")

        # Log model statistics
        log_model_stats(
            model,
            logger=self.logger,
            input_shape=(1, 3, 320, 512),
            output_dir=self.exp_manager.dirs["logs"],
            model_name=model.__class__.__name__,
        )


def main():
    """Main entry point"""
    args = parse_args()  # Assuming parse_args() is defined elsewhere
    config = TrainingConfig(args)

    # Build model and dataloaders
    model = build_network(
        n_classes=args.n_classes,
        width_list=args.width_list,
        depth_list=args.depth_list,
        head_width=args.head_width,
        head_depth=args.head_depth,
        norm=args.norm,
        act=args.act,
    )

    train_loader, val_loader = build_dataloaders(
        config=DataloaderConfig(
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            num_workers=args.num_workers,
        )
    )

    loss_fn = TPACriterion(top_k_percent=args.top_k_percent)

    # Initialize trainer and start training
    trainer = Trainer(config)
    trainer.train(model, train_loader, val_loader, loss_fn)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with improved descriptions"""
    parser = argparse.ArgumentParser(
        description="TPA Segmentation Training Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment settings
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="Unique name for this experiment run",
    )
    exp_group.add_argument(
        "--base-dir",
        type=str,
        default="experiments",
        help="Root directory for all experiments",
    )
    exp_group.add_argument(
        "--rich-logging",
        action="store_true",
        help="Enable enhanced logging with rich text formatting",
    )
    exp_group.add_argument(
        "--reset-exp",
        action="store_true",
        help="Reset experiment directory if it already exists",
    )

    # Model architecture settings
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--n-classes", type=int, default=4, help="Number of segmentation classes"
    )
    model_group.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=["background", "trailer", "trailer_bar", "trailer_ball"],
        help="Names of the classes for reporting",
    )
    model_group.add_argument(
        "--width-list",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Channel widths at each network level",
    )
    model_group.add_argument(
        "--depth-list",
        type=int,
        nargs="+",
        default=[1, 2, 2, 2, 2],
        help="Number of blocks at each network level",
    )
    model_group.add_argument(
        "--head-width", type=int, default=32, help="Number of channels in the head"
    )
    model_group.add_argument(
        "--head-depth",
        type=int,
        default=1,
        help="Number of convolution layers in the head",
    )

    # Training settings
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument(
        "--epochs", type=int, default=100, help="Total number of training epochs"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    train_group.add_argument(
        "--val-batch-size", type=int, default=1, help="Validation batch size"
    )
    train_group.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model components
    components_group = parser.add_argument_group("Model Components")
    components_group.add_argument(
        "--norm",
        type=str,
        default="bn2d",
        choices=["bn2d", "ln", "ln2d", "none"],
        help="Normalization layer type",
    )
    components_group.add_argument(
        "--act",
        type=str,
        default="relu6",
        choices=["relu6", "silu", "relu", "none"],
        help="Activation function type",
    )

    # Optimizer settings
    optim_group = parser.add_argument_group("Optimizer Configuration")
    optim_group.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["sgd", "adam", "adamw"],
        help="Optimizer algorithm",
    )
    optim_group.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate"
    )
    optim_group.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum factor for SGD"
    )
    optim_group.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay (L2 penalty)"
    )
    optim_group.add_argument(
        "--nesterov", action="store_true", help="Enable Nesterov momentum for SGD"
    )
    optim_group.add_argument(
        "--beta1", type=float, default=0.9, help="Beta1 coefficient for Adam/AdamW"
    )
    optim_group.add_argument(
        "--beta2", type=float, default=0.999, help="Beta2 coefficient for Adam/AdamW"
    )
    optim_group.add_argument(
        "--amsgrad", action="store_true", help="Enable AMSGrad variant for Adam/AdamW"
    )

    # Learning rate scheduler settings
    scheduler_group = parser.add_argument_group("Learning Rate Scheduler")
    scheduler_group.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["poly", "cosine"],
        help="Learning rate scheduler type",
    )
    scheduler_group.add_argument(
        "--min-lr", type=float, default=0.0, help="Minimum learning rate"
    )
    scheduler_group.add_argument(
        "--warmup-steps", type=int, default=0, help="Number of warmup steps"
    )
    scheduler_group.add_argument(
        "--update-frequency",
        type=str,
        default="step",
        choices=["step", "epoch"],
        help="Learning rate update frequency",
    )
    scheduler_group.add_argument(
        "--poly-power",
        type=float,
        default=0.9,
        help="Power for polynomial decay scheduler",
    )

    # Checkpointing settings
    checkpoint_group = parser.add_argument_group("Checkpointing")
    checkpoint_group.add_argument(
        "--save-freq", type=int, default=1, help="Save checkpoint every N epochs"
    )
    checkpoint_group.add_argument(
        "--keep-last-n", type=int, default=5, help="Number of checkpoints to keep"
    )

    # Miscellaneous settings
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--log-interval", type=int, default=100, help="Print loss every N batches"
    )
    misc_group.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )
    misc_group.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    misc_group.add_argument(
        "--top_k_percent",
        type=float,
        default=0.8,
        help="Top-k percentage for TPACriterion",
    )

    args = parser.parse_args()

    # Validate device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA is not available, falling back to CPU")
        args.device = "cpu"

    assert args.n_classes == len(args.class_names), (
        f"Number of classes ({args.n_classes}) does not match "
        f"number of class names ({len(args.class_names)})"
    )

    return args


if __name__ == "__main__":
    main()
