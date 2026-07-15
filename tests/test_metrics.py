from src.ml.metrics import compute_eval_result


def test_compute_eval_result_perfect_predictions():
    y_true = ["A", "B", "A", "B"]
    y_pred = ["A", "B", "A", "B"]
    result = compute_eval_result("dummy", y_true, y_pred, ["A", "B"])
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.per_class["A"].precision == 1.0
    assert result.per_class["B"].recall == 1.0


def test_compute_eval_result_with_errors():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "B", "B", "B"]
    result = compute_eval_result("dummy", y_true, y_pred, ["A", "B"])
    assert 0.0 <= result.accuracy < 1.0
    assert result.per_class["A"].support == 2
    assert result.per_class["B"].support == 2


def test_compute_eval_result_handles_missing_class_in_predictions():
    y_true = ["A", "A", "B"]
    y_pred = ["A", "A", "A"]  # model never predicts B
    result = compute_eval_result("dummy", y_true, y_pred, ["A", "B"])
    assert result.per_class["B"].recall == 0.0
    assert result.per_class["B"].support == 1
