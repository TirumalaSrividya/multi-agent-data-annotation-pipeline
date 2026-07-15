from src.orchestrator import PipelineOrchestrator
from src.schemas import Sample


def make_pool():
    texts_labels = [
        ("The team celebrated their championship victory last night.", "Sports"),
        ("The striker scored twice in the cup final match.", "Sports"),
        ("The coach announced a new training schedule for the season.", "Sports"),
        ("The central bank raised interest rates to fight inflation.", "Business"),
        ("The company posted record quarterly revenue this year.", "Business"),
        ("Shares fell sharply after the earnings report disappointed investors.", "Business"),
        ("Scientists unveiled a new AI chip with faster processing speeds.", "Sci/Tech"),
        ("Researchers announced a breakthrough in renewable battery technology.", "Sci/Tech"),
        ("The space agency confirmed a successful satellite launch.", "Sci/Tech"),
        ("The government announced new immigration policy measures.", "World"),
        ("The prime minister resigned after weeks of protests.", "World"),
        ("United Nations officials called for a ceasefire in the region.", "World"),
    ]
    return [Sample(id=str(i), text=t, true_label=l) for i, (t, l) in enumerate(texts_labels)]


def test_full_annotation_and_training_pipeline_end_to_end(test_settings):
    orchestrator = PipelineOrchestrator(test_settings)
    pool = make_pool()

    state = orchestrator.run_annotation_pipeline(pool)

    assert len(state.labelled) > 0
    assert state.mean_confidence() >= 0  # sanity: computed without error

    # Training pipeline should run on whatever got accepted, as long as it
    # has at least 2 distinct labels (guaranteed by the varied mock pool).
    labels_present = {a.label for a in state.labelled.values()}
    if len(labels_present) >= 2:
        report, model = orchestrator.run_training_pipeline()
        assert report.best_model_name == "knn"
        assert model is not None
        assert 0.0 <= report.test_result.accuracy <= 1.0


def test_annotation_pipeline_respects_token_budget(test_settings):
    test_settings.annotation.token_budget = 200  # deliberately tiny
    orchestrator = PipelineOrchestrator(test_settings)
    pool = make_pool()

    state = orchestrator.run_annotation_pipeline(pool)
    assert orchestrator.budget_tracker.used <= 200 + test_settings.llm.max_tokens_per_call
