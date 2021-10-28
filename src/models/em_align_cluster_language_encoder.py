"""
em_align_cluster_language_encoder.py | Author : Catherine Wong.

Implements modular functions for aligning and 'compressing' natural language expressed as sequences of vectors.
"""
from src.task_loaders import *
import src.utils as utils
import src.models.model_loaders as model_loaders

LanguageEncoderRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LANGUAGE_ENCODER
]


@LanguageEncoderRegistry.register
class AlignTranslateLanguageProgramsEncoder(model_loaders.ModelLoader):
    """
    Generalized model for estimating alignments and alignment-based translation distributions (between types) from paired sequences of language and tree-structured programs.
    """

    name = "align_translate_language_encoder"  # String key for config and encoder registry.

    PROGRAM, LANGUAGE = "program", "language"

    NLTK_IBM_MODEL_1 = "nltk_ibm_model_1"
    LINEARIZE_PROGRAMS_TO_ALIGN = [NLTK_IBM_MODEL_1]

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
        source_text=LANGUAGE,
        task_ids2language_tokens=None,
        alignment_model=NLTK_IBM_MODEL_1,
    ):
        """
        Learns alignments A(P, W) between language tokens and program tokens. Estimates translation distributions p(target_token | source_token) based on alignments.

        :ret: alignments; translation_probabilities; model.
        """
        task_ids2_alignable_programs = self.get_alignable_programs(
            experiment_state, task_split, task_batch_ids, alignment_model
        )
        bitext_with_task_ids = self.get_language_programs_bitext(
            self,
            task_ids2_alignable_programs,
            task_ids2language_tokens,
            source_text,
            alignment_model,
        )

    def get_alignable_programs(
        self, experiment_state, task_split, task_batch_ids, alignment_model
    ):
        if alignment_model in self.LINEARIZE_PROGRAMS_TO_ALIGN:
            tasks = experiment_state.get_tasks_for_ids(task_split, task_batch_ids)

            def linearized_programs_for_frontier(frontier):
                return [
                    experiment_state.models[model_loaders.GRAMMAR].escape_tokens(
                        entry.tokens
                    )
                    for entry in frontier.entries
                ]

            task_ids2_alignable_programs = {
                task.name: linearized_programs_for_frontier(
                    experiment_state.task_frontiers[task_split][task]
                )
                for task in tasks
            }
            return task_ids2_alignable_programs
        else:
            raise NotImplementedError

    def get_language_programs_bitext(
        self,
        task_ids2_alignable_programs,
        task_ids2language_tokens,
        source_text=LANGUAGE,
        alignment_model=NLTK_IBM_MODEL_1,
    ):
        if alignment_model in self.LINEARIZE_PROGRAMS_TO_ALIGN:
            from nltk.translate import AlignedSent

            bitext_with_task_ids = []
            if source_text == self.PROGRAM:
                source, target = task_ids2_alignable_programs, task_ids2language_tokens
            else:
                source, target = task_ids2language_tokens, task_ids2_alignable_programs
            for task_id in source:
                import itertools

                for aligned in itertools.product(source[task_id], target[task_id]):
                    aligned_sent = AlignedSent(*aligned)
                    aligned_sent.task_id = task_id
                    bitext_with_task_ids.append(aligned_sent)
            return bitext_with_task_ids
        else:
            raise NotImplementedError
