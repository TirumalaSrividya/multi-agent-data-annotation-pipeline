from src.agents.sampler_agent import SamplerAgent
from src.schemas import Annotation, Sample, SampleStatus


def make_samples():
    return [
        Sample(id="1", text="The team won the championship match last night."),
        Sample(id="2", text="The football club celebrated a big championship win."),
        Sample(id="3", text="The central bank raised interest rates today."),
        Sample(id="4", text="Scientists announced a breakthrough in quantum computing."),
    ]


def test_select_batch_returns_requested_size():
    sampler = SamplerAgent()
    samples = make_samples()
    selected = sampler.select_batch(samples, labelled={}, all_samples_by_id={s.id: s for s in samples}, batch_size=2)
    assert len(selected) == 2
    assert all(isinstance(s, Sample) for s in selected)


def test_select_batch_prefers_novel_samples_over_near_duplicates():
    sampler = SamplerAgent()
    samples = make_samples()
    all_by_id = {s.id: s for s in samples}

    # Sample 1 is already labelled; sample 2 is a near-duplicate of sample 1.
    labelled = {
        "1": Annotation(sample_id="1", label="Sports", confidence=0.9, status=SampleStatus.ACCEPTED)
    }
    remaining = [s for s in samples if s.id != "1"]

    selected = sampler.select_batch(remaining, labelled=labelled, all_samples_by_id=all_by_id, batch_size=1)
    # The most novel remaining sample relative to sample 1 (sports) should be
    # sample 3 or 4 (business/sci-tech), not the near-duplicate sample 2.
    assert selected[0].id in {"3", "4"}


def test_select_batch_returns_all_when_batch_size_exceeds_pool():
    sampler = SamplerAgent()
    samples = make_samples()[:2]
    selected = sampler.select_batch(samples, labelled={}, all_samples_by_id={s.id: s for s in samples}, batch_size=10)
    assert len(selected) == 2


def test_select_batch_empty_pool():
    sampler = SamplerAgent()
    selected = sampler.select_batch([], labelled={}, all_samples_by_id={}, batch_size=5)
    assert selected == []
