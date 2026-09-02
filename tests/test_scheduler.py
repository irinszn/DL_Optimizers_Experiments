import warnings

import torch
import pytest

from src.config import SchedulerConfig
from src.training.scheduler import build_scheduler


def _make_optimizer(lr=0.1):
    model = torch.nn.Linear(10, 1)
    return torch.optim.SGD(model.parameters(), lr=lr)


@pytest.fixture(autouse=True)
def _suppress_scheduler_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)`")
        yield


class TestBuildScheduler:
    def test_constant_no_warmup_returns_none(self):
        """Constant schedule without warmup needs no scheduler object."""
        opt = _make_optimizer()
        cfg = SchedulerConfig(name="constant", warmup_ratio=0.0)
        result = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)
        assert result is None

    def test_constant_with_warmup(self):
        """Constant schedule with warmup should still create a warmup scheduler."""
        opt = _make_optimizer()
        cfg = SchedulerConfig(name="constant", warmup_ratio=0.2)
        sched = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)
        assert sched is not None

    def test_cosine_scheduler(self):
        """Cosine schedule should reach min_lr after all epochs."""
        opt = _make_optimizer()
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.0, min_lr_ratio=0.01)
        sched = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)
        assert sched is not None

        for _ in range(10):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr == pytest.approx(0.001, abs=1e-5)

    def test_linear_scheduler(self):
        """Linear schedule should decrease LR over epochs."""
        opt = _make_optimizer()
        cfg = SchedulerConfig(name="linear", warmup_ratio=0.0, min_lr_ratio=0.1)
        sched = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)
        assert sched is not None

        for _ in range(10):
            sched.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr < 0.1

    def test_cosine_with_warmup(self):
        """Cosine + warmup should create a valid composite scheduler."""
        opt = _make_optimizer(lr=0.1)
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.3, min_lr_ratio=0.01)
        sched = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)
        assert sched is not None

    def test_unknown_scheduler_raises(self):
        """Unknown scheduler name should raise ValueError."""
        opt = _make_optimizer()
        cfg = SchedulerConfig(name="polynomial", warmup_ratio=0.0)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)

    def test_lr_decreases_over_epochs(self):
        """Cosine LR at the last epoch should be lower than at the first."""
        opt = _make_optimizer(lr=0.1)
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.0, min_lr_ratio=0.01)
        sched = build_scheduler(opt, cfg, total_epochs=20, base_lr=0.1)

        lrs = []
        for _ in range(20):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] > lrs[-1]

    def test_cosine_reaches_min_lr(self):
        """After all epochs, LR should equal base_lr * min_lr_ratio."""
        opt = _make_optimizer(lr=0.05)
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.0, min_lr_ratio=0.1)
        sched = build_scheduler(opt, cfg, total_epochs=20, base_lr=0.05)

        for _ in range(20):
            sched.step()

        final_lr = opt.param_groups[0]["lr"]
        assert final_lr == pytest.approx(0.05 * 0.1, abs=1e-5)

    def test_cosine_lr_at_midpoint(self):
        """At T_max/2, cosine LR should be approximately (base_lr + min_lr) / 2."""
        base_lr = 0.1
        min_lr_ratio = 0.01
        total_epochs = 20
        opt = _make_optimizer(lr=base_lr)
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.0, min_lr_ratio=min_lr_ratio)
        sched = build_scheduler(opt, cfg, total_epochs=total_epochs, base_lr=base_lr)

        for _ in range(total_epochs // 2):
            sched.step()

        mid_lr = opt.param_groups[0]["lr"]
        expected_mid = (base_lr + base_lr * min_lr_ratio) / 2
        assert mid_lr == pytest.approx(expected_mid, abs=1e-4)

    def test_warmup_increases_lr(self):
        """During warmup, LR should increase from near-zero toward base_lr."""
        opt = _make_optimizer(lr=0.1)
        cfg = SchedulerConfig(name="cosine", warmup_ratio=0.3, min_lr_ratio=0.01)
        sched = build_scheduler(opt, cfg, total_epochs=10, base_lr=0.1)

        lrs = []
        for _ in range(3):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] == pytest.approx(0.1 * 1e-3, abs=1e-4)
        assert lrs[0] < lrs[1] < lrs[2]
