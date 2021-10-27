"""test_synonym_language_encoder.py"""
from src.experiment_iterator import *
from src.test_experiment_iterator import *
from src.task_loaders import *
from src.models.model_loaders import *
from src.test_experiment_iterator import TEST_GRAPHICS_CONFIG, ExperimentState

TEST_SYNONYMS = "data/compositional_graphics/language/compositional_graphics_200/synthetic_synonyms_small.json"

import src.models.synonym_language_encoder as synonym_language_encoder

TEST_CONFIG = TEST_GRAPHICS_CONFIG

# Disable logging.
(
    TEST_CONFIG[METADATA][LOG_DIRECTORY],
    TEST_CONFIG[METADATA][EXPORT_DIRECTORY],
) = (None, None)


def _get_default_model(**kwargs):
    test_config = TEST_CONFIG
    test_experiment_state = ExperimentState(test_config)

    test_model = synonym_language_encoder.SynonymLanguageEncoder(
        experiment_state=test_experiment_state, **kwargs
    )
    return test_experiment_state, test_model


def test_generate_synomym_language_all():
    test_experiment_state, test_model = _get_default_model()

    max_samples_per_synonym = 2
    tasks_to_synonym_language = test_model.generate_synomym_language(
        test_experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        synonym_file_path=TEST_SYNONYMS,
        keep_original=True,
        max_samples_per_synonyms=max_samples_per_synonym,
    )
    for sentences in tasks_to_synonym_language.values():
        assert len(sentences) == max_samples_per_synonym


def test_generate_synonym_language_for_sentence():
    word2synonyms = {
        "test_letter": ["test_letter_A", "test_letter_B"],
        "test_number": ["test_number_1", "test_number_2"],
    }
    test_sentence = "test_letter test_number"
    test_experiment_state, test_model = _get_default_model()

    max_samples_per_synonyms = 2

    synonyms_no_original = test_model.generate_synonym_language_for_sentence(
        test_sentence,
        word2synonyms,
        keep_original=False,
        max_samples_per_synonyms=max_samples_per_synonyms,
    )
    assert len(synonyms_no_original) == max_samples_per_synonyms
    assert test_sentence not in synonyms_no_original
    for synonym_sentence in synonyms_no_original:
        for (token_a, token_b) in zip(test_sentence.split(), synonym_sentence.split()):
            assert token_a in token_b

    synonyms_with_original = test_model.generate_synonym_language_for_sentence(
        test_sentence,
        word2synonyms,
        keep_original=True,
        max_samples_per_synonyms=max_samples_per_synonyms,
    )

    assert len(synonyms_with_original) == max_samples_per_synonyms
    assert test_sentence in synonyms_with_original

    for synonym_sentence in synonyms_with_original:
        for (token_a, token_b) in zip(test_sentence.split(), synonym_sentence.split()):
            assert token_a in token_b
