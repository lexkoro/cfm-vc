import torch.optim as optim


def WarmupLR(
    optimizer,
    warmup_steps=1000,
    decay_steps=50000,
    max_lr=1e-4,
    min_lr=1e-5,
):
    # LambdaLR computes: actual_lr = base_lr * lr_lambda(step)
    # Since base_lr = max_lr, the lambda must return a multiplier in [min_lr/max_lr, 1.0]
    min_ratio = min_lr / max_lr

    def lr_lambda(step):
        s1 = warmup_steps
        s2 = warmup_steps + decay_steps
        if step < s1:
            # Quadratic warmup: min_ratio -> 1.0
            return min_ratio + (1.0 - min_ratio) * (step / s1) ** 2
        elif step < s2:
            # Linear decay: 1.0 -> min_ratio
            return 1.0 - (1.0 - min_ratio) * (step - s1) / decay_steps
        else:
            return min_ratio

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
