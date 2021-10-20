"""
gmm_language_encoder.py | Author : Catherine Wong.

Implements a generalized 'lexicon' for encoding sequences of vectors (eg. sentences with embedding representations for each token). Estimates parameters of the lexicon via single-component GMM over the embeddings; compresses sequences via MAP assignments to GMM components ('lexicon entries').
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task_loaders import *
import src.models.model_loaders as model_loaders

LanguageEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LANGUAGE_ENCODER
]


@LanguageEncoderRegistry.register
class GaussianLexiconLanguageEncoder(nn.Module, model_loaders.ModelLoader):
    """Language encoder that encodes and compresses language via assignments to MAP tokens under a reduced lexicon."""

    name = "gaussian_lexicon_language_encoder"  # String key for config and encoder registry.

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GaussianLexiconLanguageEncoder(
            experiment_state=experiment_state, **kwargs
        )

    def __init__(
        self,
        experiment_state=None,
    ):
        super().__init__()

    def optimize_model_for_frontiers(
        self,
        experiment_state,
        task_split=TRAIN,
        task_batch_ids=ALL,
        language_embedding_model=None,  # See utils.embedding
        language_embedding_model_library=None,  # See utils.embedding
        language_tokenizer_model=None,  # If none, attempts to autochoose from embedding model.
        language_tokenizer_model_library=None,  # If none, attempts to autochoose from embedding model.
        n_lexicon_entries=None,  # Number of entries = number of Gaussian components.
    ):
        """
        Learns the lexicon parameters based on embeddings of the model. Estimates lexicon entries represented as Gaussian distributions; assignment under the lexicon is based on p(x | z)p(z = k).
        """
        # Initialize the language by embedding it.
        self._frontier_language_to_embeddings(
            experiment_state=experiment_state,
            task_split=task_split,
            task_batch_ids=task_batch_ids,
            language_embedding_model=language_embedding_model,
            language_embedding_model_library=language_embedding_model_library,
            language_tokenizer_model=language_tokenizer_model,
            language_tokenizer_model_library=language_tokenizer_model_library,
        )

    def _frontier_language_to_embeddings(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        language_embedding_model,
        language_embedding_model_library,
        language_tokenizer_model,
        language_tokenizer_model_library,
    ):
        """
        :ret:
        language_embeddings: |language_dataset| x |max_token_len| x embedding_dim tensor w. embeddings for every sentence annotation for each task.
        attention_mask: |language_dataset| x |max_token_len| x embedding_dim tensor w. 1-hot embeddings indicating whether there is a padding location.
        """
        (
            task_strings,
            task_and_string2taskstringidx,
        ) = self._frontiers_to_untokenized_string_tensor(
            experiment_state, task_split, task_batch_ids
        )
        # TODO: embed them.

    def _frontiers_to_untokenized_task_strings(
        self, experiment_state, task_split, task_batch_ids
    ):
        """:ret:
        task_strings: array of |string_dataset| where string_dataset is the concatenation of all strings.

        task_and_string2taskstringidx: dict from (task_id, string) -> idx in task_strings
        """
        language_for_ids = experiment_state.get_language_for_ids(
            task_split, task_batch_ids
        )
        task_strings_idx = 0
        task_and_string2taskstringidx = dict()
        task_strings = []
        for task_idx, task_id in enumerate(task_batch_ids):
            for task_string in language_for_ids[task_idx]:
                task_and_string2taskstringidx[(task_id, task_string)] = task_strings_idx
                task_strings.append(task_string)
                task_strings_idx += 1
        return task_strings, task_and_string2taskstringidx
