"""
test_em_align_cluster_language_encoder.py | Author: Catherine Wong
"""
from src.experiment_iterator import *
from src.test_experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState

import src.models.em_align_cluster_language_encoder as em_align_cluster_language_encoder

TEST_CONFIG = TEST_GRAPHICS_CONFIG

# Disable logging.
(
    TEST_CONFIG[METADATA][LOG_DIRECTORY],
    TEST_CONFIG[METADATA][EXPORT_DIRECTORY],
) = (None, None)


def _get_default_align_model(**kwargs):
    test_config = TEST_CONFIG
    test_experiment_state = ExperimentState(test_config)
    # Initialize experiment state with ground truth programs
    test_experiment_state.initialize_ground_truth_task_frontiers(task_split="train")
    test_model = (
        em_align_cluster_language_encoder.AlignTranslateLanguageProgramsEncoder(
            experiment_state=test_experiment_state, **kwargs
        )
    )

    return test_experiment_state, test_model


def _get_default_alignable_programs_and_language(**kwargs):
    test_experiment_state, test_model = _get_default_align_model()

    alignable_programs = test_model.get_alignable_programs(
        test_experiment_state,
        task_split="train",
        task_batch_ids="all",
        alignment_model=test_model.NLTK_IBM_MODEL_1,
    )

    alignable_language = test_experiment_state.get_language_and_tasks_for_ids(
        task_split="train",
        task_ids="all",
    )

    def tokenize(sentences):
        return [s.split() for s in sentences]

    alignable_language_tokens = {
        task_id: tokenize(alignable_language[task_id]) for task_id in alignable_language
    }
    return (
        test_experiment_state,
        test_model,
        alignable_programs,
        alignable_language_tokens,
    )


def test_get_alignable_programs():
    test_experiment_state, test_model = _get_default_align_model()

    alignable_programs = test_model.get_alignable_programs(
        test_experiment_state,
        task_split="train",
        task_batch_ids="all",
        alignment_model=test_model.NLTK_IBM_MODEL_1,
    )

    for tokenized_programs in alignable_programs.values():
        assert len(tokenized_programs) == 1
        for tokens in tokenized_programs:
            assert len(tokens) > 0
            for token in tokens:
                assert token in test_experiment_state.models["grammar"].escaped_vocab


def test_get_language_programs_bitext():
    (
        test_experiment_state,
        test_model,
        alignable_programs,
        alignable_language_tokens,
    ) = _get_default_alignable_programs_and_language()

    bitext_with_task_ids = test_model.get_language_programs_bitext(
        alignable_programs, alignable_language_tokens
    )
    assert len(bitext_with_task_ids) > 0

    for aligned_sent in bitext_with_task_ids:
        assert aligned_sent.task_id in alignable_programs
