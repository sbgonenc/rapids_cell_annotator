import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_preflight_k_type_error():
    from rapids import preflight_stratified_kfold

    with pytest.raises(TypeError):
        preflight_stratified_kfold(["a", "b"], k="2")  # type: ignore[arg-type]


def test_preflight_negative_k_errors():
    from rapids import preflight_stratified_kfold

    with pytest.raises(ValueError, match="k must be >= 0"):
        preflight_stratified_kfold(["a", "b"], k=-1)


def test_preflight_k_zero_and_one_disabled():
    from rapids import preflight_stratified_kfold

    labels = ["a", "a", "b", "b", "b"]
    cc0, m0 = preflight_stratified_kfold(labels, k=0)
    cc1, m1 = preflight_stratified_kfold(labels, k=1)
    assert cc0 == {"a": 2, "b": 3}
    assert m0 == 2
    assert cc1 == {"a": 2, "b": 3}
    assert m1 == 2


def test_preflight_k_greater_than_n_errors():
    from rapids import preflight_stratified_kfold

    labels = ["a", "b"]
    with pytest.raises(ValueError, match=r"k \(3\) must be <= n_samples \(2\)"):
        preflight_stratified_kfold(labels, k=3)


def test_preflight_k_greater_than_min_class_count_errors():
    from rapids import preflight_stratified_kfold

    # min class count is 1 for labels: a=2, b=1
    labels = ["a", "a", "b"]
    with pytest.raises(ValueError, match=r"min class count \(1\)"):
        preflight_stratified_kfold(labels, k=2)


def test_preflight_valid_k_returns_counts():
    from rapids import preflight_stratified_kfold

    labels = ["a", "a", "b", "b", "b"]  # a=2, b=3; min=2
    cc, m = preflight_stratified_kfold(labels, k=2)
    assert cc == {"a": 2, "b": 3}
    assert m == 2

