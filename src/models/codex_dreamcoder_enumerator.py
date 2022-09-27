"""
codex_dreamcoder_enumeration.py | Author : Catherine Wong.


Adds a Codex layer on top of the dreamcoder enumeration model.
This allows Codex to be used not only as a grammar generator, but to filter the returned enumeration.
Utility wrapper function around the DreamCoder recognition model. Elevates common functions to be class functions and allows them to be called with an ExperimentState.
"""
import itertools
from src.models.laps_dreamcoder_recognition import LAPSDreamCoderRecognition

import src.models.model_loaders as model_loaders
from dreamcoder.recognition import RecognitionModel
from dreamcoder.enumeration import *
from src.task_loaders import *

AmortizedSynthesisModelRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.AMORTIZED_SYNTHESIS
]

# TODO: implement a model loader so that we can load it.

ACTIVATION_TANH, ACTIVATION_RELU = "tanh", "relu"


@AmortizedSynthesisModelRegistry.register
class CodexDreamCoderEnumeratorLoader(model_loaders.ModelLoader):
    name = "codex_dreamcoder_enumerator"

    def load_model(self, experiment_state):
        return CodexDreamCoderEnumerator()


class CodexDreamCoderEnumerator(LAPSDreamCoderRecognition):
    """CodexDreamCoderEnumerator"""

    DEFAULT_MAXIMUM_SAMPLE_FRONTIER = (
        50  # Maximum number of samples to take for a task.
    )

    # Contain the neural recognition model. This is re-trained each time optimize_model_for_frontiers is called.
    def __init__(self):
        self._neural_recognition_model = None

    def generate_solutions_and_samples(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        enumeration_timeout,
        maximum_frontier=LAPSDreamCoderRecognition.DEFAULT_MAXIMUM_FRONTIER,
        maximum_sample_frontier=DEFAULT_MAXIMUM_SAMPLE_FRONTIER,
        cpus=LAPSDreamCoderRecognition.DEFAULT_CPUS,
        solver=LAPSDreamCoderRecognition.DEFAULT_ENUMERATION_SOLVER,
        evaluation_timeout=LAPSDreamCoderRecognition.DEFAULT_EVALUATION_TIMEOUT,
        max_mem_per_enumeration_thread=LAPSDreamCoderRecognition.DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD,
        solver_directory=LAPSDreamCoderRecognition.DEFAULT_BINARY_DIRECTORY,
        likelihood_model=INDUCTIVE_EXAMPLES_DISCOUNTED_PRIOR_LIKELIHOOD_MODEL,
        **kwargs,
    ):
        """
        Generates samples via an enumerative PCFG for tasks.
        """
        for task_split in task_splits:
            tasks_to_attempt = experiment_state.get_tasks_for_ids(
                task_split=task_split,
                task_ids=task_ids_in_splits[task_split],
                include_samples=False,
            )

            # Enumerate frontiers from the PCFG.
            (
                pcfg_sample_frontiers,
                _,
            ) = self._neural_recognition_model.enumerateFrontiers(
                tasks=tasks_to_attempt,
                maximumFrontier=maximum_sample_frontier,
                enumerationTimeout=enumeration_timeout,
                CPUs=cpus,
                solver=solver,
                evaluationTimeout=evaluation_timeout,
                max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
                solver_directory=solver_directory,
                testing=task_split == TEST,
                likelihood_model=likelihood_model,
            )

            for pcfg_sample_frontier in pcfg_sample_frontiers:
                # Get Codex likelihood.
                codex_frontier = self.codex_likelihood_for_frontier(
                    pcfg_sample_frontier.topK(maximum_sample_frontier)
                )
                # Combine specs?

                # Rethreshold the frontier.

            # Update the frontiers.

    def codex_likelihood_for_frontier(self, frontier):
        import pdb

        pdb.set_trace()
        # Are we evaluating the posthoc likelihood of a bunch of frontiers?
