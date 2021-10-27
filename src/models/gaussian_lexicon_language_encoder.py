"""
gmm_language_encoder.py | Author : Catherine Wong.

Implements a generalized 'lexicon' for encoding sequences of vectors (eg. sentences with embedding representations for each token). Estimates parameters of the lexicon via single-component GMM over the embeddings; compresses sequences via MAP assignments to GMM components ('lexicon entries').

TODO:
    Investigate GMM initialization (seems to converge pseudorandomly.)
    Investigate behavior due to high dimensions - low-rank approximation?
    Investigate clustering (visual inspection).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from src.gmm_torch.gmm import GaussianMixture, COVARIANCE_DIAGONAL, COVARIANCE_FULL

from src.task_loaders import *
import src.utils as utils
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
        language_embedding_type=None,  # See utils.embedding
        n_lexicon_entries=None,  # Number of entries = number of Gaussian components.
        covariance_type=COVARIANCE_DIAGONAL,
        max_em_iter=100,
    ):
        """
        Learns the lexicon parameters based on embeddings of the model. Estimates lexicon entries represented as Gaussian distributions; assignment under the lexicon is based on p(x | z)p(z = k).
        """
        # Initialize the language by embedding it.
        (
            language_embeddings,
            attention_mask,
            tokens_tensor,
            task_and_string2taskstringidx,
        ) = self._frontier_language_to_embeddings(
            experiment_state=experiment_state,
            task_split=task_split,
            task_batch_ids=task_batch_ids,
            language_embedding_type=language_embedding_type,
        )
        print(
            f"\tInitialized language embeddings for {len(tokens_tensor)} strings; max_len: {language_embeddings.size()[1]}."
        )

        gmm_lexicon = self._init_gmm_lexicon(
            language_embeddings, n_lexicon_entries, covariance_type=covariance_type
        )

        (
            gmm_lexicon,
            unraveled_nonzero_embeddings,
            interpretable_lexicon_names,
        ) = self._fit_gmm_lexicon(
            language_embeddings,
            attention_mask,
            tokens_tensor,
            gmm_lexicon,
            max_em_iter,
        )

        (
            original_tokens_language_for_tasks,
            compressed_lexicon_language_for_tasks,
            compressed_interpretable_language_for_tasks,
        ) = self.compress_frontier_strings_map_assignments_to_lexicon(
            experiment_state,
            task_split,
            task_batch_ids,
            task_and_string2taskstringidx,
            gmm_lexicon,
            unraveled_nonzero_embeddings,
            tokens_tensor,
            interpretable_lexicon_names,
        )

        # Set internal.
        self.gmm_lexicon = gmm_lexicon
        self.interpretable_lexicon_names = interpretable_lexicon_names
        self.original_tokens_language_for_tasks = original_tokens_language_for_tasks
        self.compressed_lexicon_language_for_tasks = (
            compressed_lexicon_language_for_tasks
        )
        self.compressed_interpretable_language_for_tasks = (
            compressed_interpretable_language_for_tasks
        )

        return (
            gmm_lexicon,
            interpretable_lexicon_names,
            original_tokens_language_for_tasks,
            compressed_lexicon_language_for_tasks,
            compressed_interpretable_language_for_tasks,
        )

    def visualize_lexicon_and_map_assignments_to_lexicon(
        self,
        experiment_state,  # To fit model_fn API
        task_split=TRAIN,  # To fit model_fn API
        task_batch_ids=ALL,  # To fit model_fn API
        gmm_lexicon=None,
        interpretable_lexicon_names=None,
        original_tokens_language_for_tasks=None,
        compressed_lexicon_language_for_tasks=None,
        compressed_interpretable_language_for_tasks=None,
    ):
        if gmm_lexicon is None:
            gmm_lexicon = self.gmm_lexicon
            interpretable_lexicon_names = self.interpretable_lexicon_names
            original_tokens_language_for_tasks = self.original_tokens_language_for_tasks
            compressed_lexicon_language_for_tasks = (
                self.compressed_lexicon_language_for_tasks
            )
            compressed_interpretable_language_for_tasks = (
                self.compressed_interpretable_language_for_tasks
            )

        # TODO(@catwong): TSNE / visualize GMM
        # Report the components
        print(
            f"GMM Lexicon with {gmm_lexicon.n_components} w. mean-dimension {gmm_lexicon.n_features}."
        )
        print("Interpretable component names: ")
        print("\n\t" + "\n\t".join(interpretable_lexicon_names))

        # Report the rewritten language
        print(
            f"Rewritten frontiers for {len(original_tokens_language_for_tasks)} tasks. O: original; C: compressed."
        )
        for task_id in original_tokens_language_for_tasks:
            print(f"Task: {task_id}")
            for sentence_idx, original_tokens in enumerate(
                original_tokens_language_for_tasks[task_id]
            ):
                compressed_tokens = compressed_interpretable_language_for_tasks[
                    task_id
                ][sentence_idx]
                print("\tO:" + " ".join(original_tokens))
                print("\tC:" + " ".join(compressed_tokens))
                print("\n")
            print("========")

    def compress_frontier_strings_map_assignments_to_lexicon(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        task_and_string2taskstringidx,
        gmm_lexicon,
        unraveled_nonzero_embeddings,
        tokens_tensor,
        interpretable_lexicon_names,
    ):
        global_idx = 0
        compressed_lexicon_tokens_tensor = []
        compressed_interpretable_tokens_tensor = []
        map_unraveled_tokens = gmm_lexicon.predict(unraveled_nonzero_embeddings)

        for sentence in tokens_tensor:
            compressed_token_sentence = []
            compressed_interpretable_tokens_sentence = []
            for token in sentence:
                map_lexicon_token = map_unraveled_tokens[global_idx]
                map_interpretable_token = interpretable_lexicon_names[map_lexicon_token]
                compressed_token_sentence.append(map_lexicon_token)
                compressed_interpretable_tokens_sentence.append(map_interpretable_token)

                global_idx += 1
            compressed_lexicon_tokens_tensor.append(compressed_token_sentence)
            compressed_interpretable_tokens_tensor.append(
                compressed_interpretable_tokens_sentence
            )

        # Finally, rewrite the originals.
        original_tokens_language_for_tasks = defaultdict(list)
        compressed_lexicon_language_for_tasks = defaultdict(list)
        compressed_interpretable_language_for_tasks = defaultdict(list)
        for task_id in task_batch_ids:
            for str_idx, _ in enumerate(
                experiment_state.task_language[task_split][task_id]
            ):
                global_idx = task_and_string2taskstringidx[(task_id, str_idx)]

                tokens = tokens_tensor[global_idx]
                compressed_tokens = compressed_lexicon_tokens_tensor[global_idx]
                compressed_interpretable = compressed_interpretable_tokens_tensor[
                    global_idx
                ]

                original_tokens_language_for_tasks[task_id].append(tokens)
                compressed_lexicon_language_for_tasks[task_id].append(compressed_tokens)
                compressed_interpretable_language_for_tasks[task_id].append(
                    compressed_interpretable
                )
        return (
            original_tokens_language_for_tasks,
            compressed_lexicon_language_for_tasks,
            compressed_interpretable_language_for_tasks,
        )

    def _init_gmm_lexicon(
        self,
        language_embeddings,
        n_lexicon_entries,
        covariance_type=COVARIANCE_DIAGONAL,
    ):
        gmm_lexicon = GaussianMixture(
            n_components=n_lexicon_entries,
            n_features=language_embeddings.size()[-1],
            covariance_type=covariance_type,
        )
        return gmm_lexicon

    def _fit_gmm_lexicon(
        self,
        language_embeddings,
        attention_mask,
        tokens_tensor,
        gmm_lexicon,
        max_em_iter,
    ):

        # Unravel language_embeddings into a |B * tokens| x |embedding_dim| tensor of all of the flattened strings for fitting. TODO (catwong): remove padding after unraveling if GMM doesn't converge.
        unraveled_embeddings = torch.flatten(
            language_embeddings, start_dim=0, end_dim=1
        )
        unraveled_tokens_mask = (
            ~attention_mask.flatten()
        )  # Unravel and take where there are nonzero tokens
        unraveled_nonzero_embeddings = unraveled_embeddings[unraveled_tokens_mask]
        gmm_lexicon.fit(unraveled_embeddings, n_iter=max_em_iter)

        interpretable_lexicon_names = self._get_interpretable_lexicon_names(
            gmm_lexicon, unraveled_nonzero_embeddings, tokens_tensor
        )
        return gmm_lexicon, unraveled_nonzero_embeddings, interpretable_lexicon_names

    def _get_interpretable_lexicon_names(
        self, gmm_lexicon, unraveled_nonzero_embeddings, tokens_tensor
    ):
        """
        :ret: lexicon_names: |lexicon_size| list where lexicon_names[i] = ith word-type of the highest probability lexical item for that component.
        """
        component_log_probs = gmm_lexicon._estimate_log_prob(
            unraveled_nonzero_embeddings
        )
        argmax_indices = torch.argmax(component_log_probs, 0).squeeze().tolist()
        flattened_tokens_tensor = [
            item for sublist in tokens_tensor for item in sublist
        ]
        interpretable_names = [
            flattened_tokens_tensor[argmax_idx] for argmax_idx in argmax_indices
        ]
        return interpretable_names

    def _frontier_language_to_embeddings(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        language_embedding_type,
    ):
        """
        :params:
            language_embedding_type: see utils.untokenized_strings_to_pretrained_embeddings for types.
        :ret:
        language_embeddings: |language_dataset| x |max_token_len| x embedding_dim tensor w. embeddings for every sentence annotation for each task.

        attention_mask: |language_dataset| x |max_token_len| x embedding_dim tensor w. 1-hot embeddings indicating whether there is a padding location.

        tokens_tensor: |language_dataset| x unpadded lists of string tokens
        task_and_string2taskstringidx: {(task_id, string) : row_idx into language_embeddings}

        """
        (
            strings_tensor,
            task_and_string2taskstringidx,
        ) = self._frontiers_to_untokenized_task_strings(
            experiment_state, task_split, task_batch_ids
        )
        (
            tokens_tensor,
            padded_token_embeddings,
            attention_mask,
        ) = utils.untokenized_strings_to_pretrained_embeddings(
            strings_tensor, language_embedding_type
        )
        return (
            padded_token_embeddings,
            attention_mask,
            tokens_tensor,
            task_and_string2taskstringidx,
        )

    def _frontiers_to_untokenized_task_strings(
        self, experiment_state, task_split, task_batch_ids
    ):
        """:ret:
        task_strings: array of |string_dataset| where string_dataset is the concatenation of all strings.

        task_and_string2taskstringidx: dict from (task_id, string_id_for_task) -> global idx in task_strings
        """
        language_for_ids = experiment_state.get_language_for_ids(
            task_split, task_batch_ids
        )
        task_strings_idx = 0
        task_and_string2taskstringidx = dict()
        task_strings = []
        for task_idx, task_id in enumerate(task_batch_ids):
            for string_id_for_task, task_string in enumerate(
                language_for_ids[task_idx]
            ):
                task_and_string2taskstringidx[
                    (task_id, string_id_for_task)
                ] = task_strings_idx
                task_strings.append(task_string)
                task_strings_idx += 1
        return task_strings, task_and_string2taskstringidx
