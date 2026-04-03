from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR
from functools import partial
import math


# ── Scheduler factories ──────────────────────────────────────────────────────

def _identity_lr_lambda(_):
    return 1.0


def _warmup_lr_lambda(step, learning_rate, warmup_steps=1000):
    step = max(0, int(step))
    warmup_steps = max(2, int(warmup_steps))
    if step + 1 >= warmup_steps:
        return learning_rate
    return learning_rate * (math.log(step + 2) / math.log(warmup_steps))

def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """Warm up the learning rate to args.learning_rate over early steps."""
    return LambdaLR(
        optimizer,
        lr_lambda=partial(
            _warmup_lr_lambda,
            learning_rate=getattr(args, "learning_rate", 1e-3),
            warmup_steps=getattr(args, "lr_step_size", 1000),
        ),
    )


def none_scheduler(optimizer, args):
    """No-op scheduler used by the notebook's vanilla SGD recipe."""
    return LambdaLR(optimizer, lr_lambda=_identity_lr_lambda)


# ── Registry ─────────────────────────────────────────────────────────────────

schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}
