import functools

from skrl.resources.schedulers.jax import KLAdaptiveLR, is_kl_adaptive, kl_adaptive


def test_kl_adaptive_schedule_is_marked():
    assert is_kl_adaptive(KLAdaptiveLR())
    assert is_kl_adaptive(kl_adaptive(kl_threshold=0.01))
    assert is_kl_adaptive(functools.partial(KLAdaptiveLR, kl_threshold=0.01)())
    assert is_kl_adaptive((lambda **kwargs: KLAdaptiveLR(**kwargs))(min_lr=0.001))


def test_custom_schedules():
    def schedule(count):
        return 1.0

    class CustomKLAdaptive:
        kl_adaptive = True

        def __call__(self, count, *, lr=None, kl=None):
            return 1.0 if lr is None else lr

    assert not is_kl_adaptive(schedule)
    assert not is_kl_adaptive(KLAdaptiveLR)  # factory, not a constructed schedule
    assert is_kl_adaptive(CustomKLAdaptive())
    schedule.kl_adaptive = "yes"  # only ``True`` opts in
    assert not is_kl_adaptive(schedule)
