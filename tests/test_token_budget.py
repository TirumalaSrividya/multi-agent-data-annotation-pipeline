from src.utils.token_budget import TokenBudgetTracker


def test_has_budget_true_initially():
    tracker = TokenBudgetTracker(budget=1000)
    assert tracker.has_budget(500)
    assert tracker.remaining == 1000


def test_consume_reduces_remaining():
    tracker = TokenBudgetTracker(budget=1000)
    tracker.consume(300)
    assert tracker.used == 300
    assert tracker.remaining == 700


def test_has_budget_false_when_exhausted():
    tracker = TokenBudgetTracker(budget=100)
    tracker.consume(100)
    assert not tracker.has_budget(1)
    assert tracker.remaining == 0


def test_consume_negative_raises():
    tracker = TokenBudgetTracker(budget=100)
    try:
        tracker.consume(-5)
        assert False, "expected ValueError"
    except ValueError:
        pass
