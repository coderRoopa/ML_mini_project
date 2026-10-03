import numpy as np

from src.utils import pick_threshold, sens_spec_pred


def test_sens_spec():
    y = np.array([1, 1, 0, 0])
    sens, spec = sens_spec_pred(y, np.array([True, False, False, True]))
    assert sens == 0.5 and spec == 0.5


def test_pick_threshold_reaches_target():
    y = np.array([1] * 50 + [0] * 50)
    p = np.concatenate([np.linspace(0.4, 0.95, 50), np.linspace(0.05, 0.6, 50)])
    t = pick_threshold(y, p, 0.95)
    assert sens_spec_pred(y, p >= t)[0] >= 0.95
    assert 0.05 <= t <= 0.95
