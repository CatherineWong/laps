"""
test_gaussian_lexicon_language_encoder.py
"""
from src.experiment_iterator import *
from src.test_experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState
import src.models.gaussian_lexicon_language_encoder as gaussian_lexicon_language_encoder

TEST_CONFIG = TEST_GRAPHICS_CONFIG

# Disable logging.
(
    TEST_CONFIG[METADATA][LOG_DIRECTORY],
    TEST_CONFIG[METADATA][EXPORT_DIRECTORY],
) = (None, None)


def _get_default_gaussian_encoder(**kwargs):
    test_config = TEST_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_model = gaussian_lexicon_language_encoder.GaussianLexiconLanguageEncoder(
        experiment_state=test_experiment_state, **kwargs
    )
    return test_experiment_state, test_model


def test_frontiers_to_untokenized_string_list():
    test_task_ids = ["a small triangle", "a medium triangle"]
    test_experiment_state, test_model = _get_default_gaussian_encoder()

    (
        task_strings,
        task_and_string2taskstringidx,
    ) = test_model._frontiers_to_untokenized_task_strings(
        test_experiment_state, task_split="train", task_batch_ids=test_task_ids
    )

    assert len(task_strings) == len(task_and_string2taskstringidx)
