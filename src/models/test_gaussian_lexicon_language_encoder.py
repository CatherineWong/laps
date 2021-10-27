"""
test_gaussian_lexicon_language_encoder.py
"""
from src.experiment_iterator import *
from src.test_experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState

import src.utils as utils
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


def test_frontier_language_to_embeddings():
    test_task_ids = ["a small triangle", "a medium triangle"]
    test_experiment_state, test_model = _get_default_gaussian_encoder()

    (
        language_embeddings,
        attention_mask,
        tokens_tensor,
        task_and_string2taskstringidx,
    ) = test_model._frontier_language_to_embeddings(
        test_experiment_state,
        task_split="train",
        task_batch_ids=test_task_ids,
        language_embedding_type=utils.EMBEDDING_STATIC,
    )
    num_strings = len(task_and_string2taskstringidx)
    assert language_embeddings.size()[0] == num_strings
    assert attention_mask.size()[0] == num_strings
    assert len(tokens_tensor) == num_strings

    assert language_embeddings.size()[1] == attention_mask.size()[1]


def test_init_gmm_lexicon():
    test_task_ids = ["a small triangle", "a medium triangle"]
    test_experiment_state, test_model = _get_default_gaussian_encoder()

    (
        language_embeddings,
        attention_mask,
        tokens_tensor,
        task_and_string2taskstringidx,
    ) = test_model._frontier_language_to_embeddings(
        test_experiment_state,
        task_split="train",
        task_batch_ids=test_task_ids,
        language_embedding_type=utils.EMBEDDING_STATIC,
    )

    TEST_LEXICON_SIZE = 3

    gmm_lexicon = test_model._init_gmm_lexicon(language_embeddings, TEST_LEXICON_SIZE)

    assert gmm_lexicon.n_components == TEST_LEXICON_SIZE
    assert gmm_lexicon.n_features == language_embeddings.size()[-1]


def test_fit_gmm_lexicon():
    """
    Integration test.
    """
    test_task_ids = ["a small triangle", "a medium triangle"]
    test_experiment_state, test_model = _get_default_gaussian_encoder()

    (
        language_embeddings,
        attention_mask,
        tokens_tensor,
        task_and_string2taskstringidx,
    ) = test_model._frontier_language_to_embeddings(
        test_experiment_state,
        task_split="train",
        task_batch_ids=test_task_ids,
        language_embedding_type=utils.EMBEDDING_STATIC,
    )

    TEST_LEXICON_SIZE = 3

    gmm_lexicon = test_model._init_gmm_lexicon(language_embeddings, TEST_LEXICON_SIZE)

    test_model._fit_gmm_lexicon(
        language_embeddings, attention_mask, tokens_tensor, gmm_lexicon, max_em_iter=2
    )


def test_optimize_model_for_frontiers():
    test_task_ids = ["a small triangle", "a medium triangle"]

    test_experiment_state, test_model = _get_default_gaussian_encoder()

    TEST_LEXICON_SIZE = 3
    MAX_EM_ITER = 10

    test_model.optimize_model_for_frontiers(
        experiment_state=test_experiment_state,
        task_split="train",
        task_batch_ids=test_task_ids,
        language_embedding_type=utils.EMBEDDING_STATIC,
        n_lexicon_entries=TEST_LEXICON_SIZE,
        max_em_iter=MAX_EM_ITER,
    )


def test_visualize_lexicon_and_map_assignments_to_lexicon():
    test_task_ids = ["a small triangle", "a medium triangle"]

    test_experiment_state, test_model = _get_default_gaussian_encoder()

    TEST_LEXICON_SIZE = 5
    MAX_EM_ITER = 100

    (
        gmm_lexicon,
        interpretable_lexicon_names,
        original_tokens_language_for_tasks,
        compressed_lexicon_language_for_tasks,
        compressed_interpretable_language_for_tasks,
    ) = test_model.optimize_model_for_frontiers(
        experiment_state=test_experiment_state,
        task_split="train",
        task_batch_ids=test_task_ids,
        language_embedding_type=utils.EMBEDDING_STATIC,
        n_lexicon_entries=TEST_LEXICON_SIZE,
        max_em_iter=MAX_EM_ITER,
    )
    test_model.visualize_lexicon_and_maps_assignments_to_lexicon(
        None,
        None,
        None,
        gmm_lexicon,
        interpretable_lexicon_names,
        original_tokens_language_for_tasks,
        compressed_lexicon_language_for_tasks,
        compressed_interpretable_language_for_tasks,
    )
