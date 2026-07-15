from src.agents.quality_assessor_agent import QualityAssessorAgent
from src.schemas import Annotation, Sample, SampleStatus
from src.utils.token_budget import TokenBudgetTracker


def make_qa(mock_llm_client, label_set, threshold=0.8):
    return QualityAssessorAgent(
        mock_llm_client, label_set, task_name="news_topic_classification", low_confidence_threshold=threshold
    )


def test_filter_low_confidence_splits_correctly(mock_llm_client, label_set):
    qa = make_qa(mock_llm_client, label_set)
    annotations = [
        Annotation(sample_id="1", label="Sports", confidence=0.9),
        Annotation(sample_id="2", label="World", confidence=0.4),
        Annotation(sample_id="3", label="Business", confidence=0.79),
    ]
    low, high = qa.filter_low_confidence(annotations)
    assert {a.sample_id for a in low} == {"2", "3"}
    assert {a.sample_id for a in high} == {"1"}


def test_review_batch_updates_confidence_and_returns_two_buckets(mock_llm_client, label_set):
    qa = make_qa(mock_llm_client, label_set)
    sample = Sample(id="1", text="The central bank raised interest rates today.")
    annotation = Annotation(sample_id="1", label="Business", confidence=0.3, status=SampleStatus.UNDER_REVIEW)

    budget = TokenBudgetTracker(budget=100000)
    accepted, still_low = qa.review_batch([annotation], {"1": sample}, budget)

    assert len(accepted) + len(still_low) == 1
    reviewed = (accepted + still_low)[0]
    assert reviewed.review_count == 1
    assert reviewed.source == "quality_assessor"
    assert 0.0 <= reviewed.confidence <= 1.0


def test_review_batch_respects_token_budget(mock_llm_client, label_set):
    qa = make_qa(mock_llm_client, label_set)
    sample = Sample(id="1", text="Some ambiguous text here.")
    annotation = Annotation(sample_id="1", label="World", confidence=0.2)

    budget = TokenBudgetTracker(budget=0)  # no budget left
    accepted, still_low = qa.review_batch([annotation], {"1": sample}, budget)
    assert accepted == []
    assert len(still_low) == 1
    assert still_low[0].review_count == 0  # was deferred, not actually reviewed
