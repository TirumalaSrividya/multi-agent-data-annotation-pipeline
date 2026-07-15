from src.agents.annotator_agent import AnnotatorAgent
from src.schemas import Sample, SampleStatus
from src.utils.token_budget import TokenBudgetTracker


def test_annotate_batch_produces_valid_labels(mock_llm_client, label_set):
    agent = AnnotatorAgent(mock_llm_client, label_set, task_name="news_topic_classification")
    samples = [
        Sample(id="1", text="The team scored a last-minute goal to win the match."),
        Sample(id="2", text="The company reported record quarterly profits."),
    ]
    budget = TokenBudgetTracker(budget=100000)
    annotations = agent.annotate_batch(samples, budget)

    assert len(annotations) == 2
    for a in annotations:
        assert a.label in label_set
        assert 0.0 <= a.confidence <= 1.0
        assert a.status == SampleStatus.ANNOTATED
        assert a.tokens_used > 0


def test_annotate_batch_stops_at_token_budget(mock_llm_client, label_set):
    agent = AnnotatorAgent(mock_llm_client, label_set, task_name="news_topic_classification", max_tokens_per_call=50)
    samples = [Sample(id=str(i), text=f"Sample text number {i} about various topics.") for i in range(20)]

    # Budget only large enough for a couple of calls.
    budget = TokenBudgetTracker(budget=120)
    annotations = agent.annotate_batch(samples, budget)

    assert len(annotations) < len(samples)
    assert budget.used <= budget.budget + 60  # allow the one call that pushed it over


def test_annotate_batch_empty_input(mock_llm_client, label_set):
    agent = AnnotatorAgent(mock_llm_client, label_set, task_name="news_topic_classification")
    budget = TokenBudgetTracker(budget=1000)
    assert agent.annotate_batch([], budget) == []
