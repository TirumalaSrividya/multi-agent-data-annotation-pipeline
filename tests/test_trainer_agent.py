import random

from src.agents.trainer_agent import TrainerAgent
from src.config import TrainerConfig
from src.schemas import TrainingReport


def make_synthetic_dataset(n_per_class=30):
    rng = random.Random(0)
    templates = {
        "Sports": ["the team won the {n} match", "player scored in the {n} game"],
        "Business": ["the company reported {n} profits", "stocks rose by {n} percent"],
    }
    texts, labels = [], []
    for label, tpls in templates.items():
        for i in range(n_per_class):
            t = rng.choice(tpls).format(n=i)
            texts.append(t)
            labels.append(label)
    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


def test_split_produces_nonempty_splits_and_preserves_total():
    config = TrainerConfig(candidate_models=["knn"], train_split=0.7, eval_split=0.15, test_split=0.15,
                            target_accuracy=0.5, max_epochs=2, early_stopping_patience=1, batch_size=8)
    agent = TrainerAgent(label_set=["Sports", "Business"], trainer_config=config, random_seed=1)
    texts, labels = make_synthetic_dataset(n_per_class=20)

    train_t, train_l, eval_t, eval_l, test_t, test_l = agent.split(texts, labels)

    assert len(train_t) + len(eval_t) + len(test_t) == len(texts)
    assert len(eval_t) > 0
    assert len(test_t) > 0
    assert set(train_l) <= {"Sports", "Business"}


def test_run_trains_knn_and_produces_training_report():
    config = TrainerConfig(candidate_models=["knn"], train_split=0.7, eval_split=0.15, test_split=0.15,
                            target_accuracy=0.3, max_epochs=2, early_stopping_patience=1, batch_size=8)
    agent = TrainerAgent(label_set=["Sports", "Business"], trainer_config=config, random_seed=1)
    texts, labels = make_synthetic_dataset(n_per_class=30)

    report, model = agent.run(texts, labels)

    assert isinstance(report, TrainingReport)
    assert report.best_model_name == "knn"
    assert 0.0 <= report.test_result.accuracy <= 1.0
    assert set(report.test_result.per_class.keys()) == {"Sports", "Business"}
    assert model is not None


def test_run_raises_with_single_label():
    config = TrainerConfig(candidate_models=["knn"])
    agent = TrainerAgent(label_set=["Sports"], trainer_config=config)
    try:
        agent.run(["text a", "text b"], ["Sports", "Sports"])
        assert False, "expected ValueError"
    except ValueError:
        pass
