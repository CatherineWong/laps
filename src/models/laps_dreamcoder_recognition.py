"""
laps_dreamcoder_recognition.py | Author : Catherine Wong.

Utility wrapper function around the DreamCoder recognition model. Elevates common functions to be class functions and allows them to be called with an ExperimentState.
"""
import itertools

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
class LAPSDreamCoderRecognitionLoader(model_loaders.ModelLoader):
    name = "dreamcoder_recognition"

    def load_model(self, experiment_state):
        return LAPSDreamCoderRecognition()


class LAPSDreamCoderRecognition:
    """LAPSDreamCoderRecognition: containiner wrapper for a DreamCoder recognition model. The neural weights are fully reset and retrained when optimize_model_for_frontiers is called."""

    DEFAULT_MAXIMUM_FRONTIER = 5  # Maximum top-programs to keep in frontier
    DEFAULT_MAXIMUM_SAMPLE_FRONTIER = (
        100  # Maximum number of samples to take for a task.
    )
    DEFAULT_CPUS = 12  # Parallel CPUs
    DEFAULT_ENUMERATION_SOLVER = "ocaml"  # OCaml, PyPy, or Python enumeration
    DEFAULT_SAMPLER = "helmholtz"
    DEFAULT_BINARY_DIRECTORY = os.path.join(
        DEFAULT_ENUMERATION_SOLVER, "bin"
    )  # Assumes you're almost definitely running this on a linux machine.
    DEFAULT_EVALUATION_TIMEOUT = 1  # Timeout for evaluating a program on a task
    DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD = 1000000000  # Max memory usage per thread

    # Contain the neural recognition model. This is re-trained each time optimize_model_for_frontiers is called.
    def __init__(self):
        self._neural_recognition_model = None

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        enumeration_timeout,
        maximum_frontier=DEFAULT_MAXIMUM_FRONTIER,
        cpus=DEFAULT_CPUS,
        solver=DEFAULT_ENUMERATION_SOLVER,
        evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT,
        max_mem_per_enumeration_thread=DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD,
        solver_directory=DEFAULT_BINARY_DIRECTORY,
        likelihood_model=INDUCTIVE_EXAMPLES_LIKELIHOOD_MODEL,
        **kwargs,
    ):
        """
        Infers programs for tasks via top-down enumerative search from the grammar.
        Updates Frontiers in experiment_state with discovered programs.

        Wrapper function around recognition.enumerateFrontiers from dreamcoder.recognition.
        """
        for task_split in task_splits:
            tasks_to_attempt = experiment_state.get_tasks_for_ids(
                task_split=task_split,
                task_ids=task_ids_in_splits[task_split],
                include_samples=False,
            )
            new_frontiers, _ = self._neural_recognition_model.enumerateFrontiers(
                tasks=tasks_to_attempt,
                maximumFrontier=maximum_frontier,
                enumerationTimeout=enumeration_timeout,
                CPUs=cpus,
                solver=solver,
                evaluationTimeout=evaluation_timeout,
                max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
                solver_directory=solver_directory,
                testing=task_split == TEST,
                likelihood_model=likelihood_model,
            )

            experiment_state.update_frontiers(
                new_frontiers=new_frontiers,
                maximum_frontier=maximum_frontier,
                task_split=task_split,
                is_sample=False,
                report_frontiers=True,
            )

    def optimize_model_for_frontiers(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        sample_training_ratio=0.0,  # How often to try to train on existing samples.
        task_encoder_types=[
            model_loaders.EXAMPLES_ENCODER
        ],  # Task encoders to use: [EXAMPLES_ENCODER, LANGUAGE_ENCODER]
        recognition_train_steps=10000,  # Gradient steps to train model.
        recognition_train_timeout=None,  # Alternatively, how long to train the model
        recognition_train_epochs=None,  # Alternatively, how many epochs to train
        sample_evaluation_timeout=1.0,  # How long to spend trying to evaluate samples.
        matrix_rank=None,  # Maximum rank of bigram transition matrix for contextual recognition model. Defaults to full rank.
        mask=False,  # Unconditional bigram masking
        activation=ACTIVATION_TANH,
        contextual=True,
        bias_optimal=True,
        auxiliary_loss=True,
        cuda=False,
        cpus=12,
        max_mem_per_enumeration_thread=1000000,
        require_ground_truth_frontiers=False,
        **kwargs,
    ):
        """Trains a new recognition model with respect to the frontiers. Updates the experiment_state.models[AMORTIZED_SYNTHESIS] to contain the trained model."""

        # Skip training if no non-empty frontiers.
        frontiers_in_splits = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=False,
        )
        solved_frontiers = list(itertools.chain(*frontiers_in_splits.values()))

        # Further, try and score the frontiers so that we can train with them.

        if require_ground_truth_frontiers and len(solved_frontiers) < 1:
            print(
                f"require_ground_truth_frontiers=True and no non-empty frontiers. skipping optimize_model_for_frontiers"
            )
            return
        # Initialize specification encoders.
        example_encoder = self._maybe_initialize_example_encoder(
            task_encoder_types, experiment_state
        )
        language_encoder = self._maybe_initialize_language_encoder(
            task_encoder_types, experiment_state
        )
        # Initialize the neural recognition model.
        self._neural_recognition_model = RecognitionModel(
            example_encoder=example_encoder,
            language_encoder=language_encoder,
            grammar=experiment_state.models[model_loaders.GRAMMAR],
            mask=mask,
            rank=matrix_rank,
            activation=activation,
            cuda=cuda,
            contextual=contextual,
            pretrained_model=None,
            helmholtz_nearest_language=0,
            helmholtz_translations=None,  # This object contains information for using the joint generative model over programs and language. We would only use this if we had a translation model for the samples
            nearest_encoder=None,
            nearest_tasks=[],
            id=0,
        )

        # Train the model. We no longer take online samples from the grammar during training.

        # Returns any existing samples in the experiment state
        def get_sample_frontiers():
            return experiment_state.get_frontiers_for_ids(
                task_split=TRAIN,
                task_ids=[],
                include_samples=True,
                include_ground_truth_tasks=False,
            )

        self._neural_recognition_model.train(
            solved_frontiers,
            biasOptimal=bias_optimal,
            helmholtzFrontiers=get_sample_frontiers(),
            CPUs=cpus,
            evaluationTimeout=sample_evaluation_timeout,
            timeout=recognition_train_timeout,
            steps=recognition_train_steps,
            helmholtzRatio=sample_training_ratio,
            auxLoss=auxiliary_loss,
            vectorized=True,
            epochs=recognition_train_epochs,
            generateNewHelmholtz=False,  # Disallow generating new samples within the model.
        )

    def _maybe_initialize_example_encoder(self, task_encoder_types, experiment_state):
        if model_loaders.EXAMPLES_ENCODER not in task_encoder_types:
            return None
        # Initialize from tasks.
        model_initializer_fn = experiment_state.models[model_loaders.EXAMPLES_ENCODER]
        return model_initializer_fn(experiment_state)

    def _maybe_initialize_language_encoder(self, task_encoder_types, experiment_state):
        if model_loaders.LANGUAGE_ENCODER not in task_encoder_types:
            return None
        # Initialize from tasks.
        model_initializer_fn = experiment_state.models[model_loaders.LANGUAGE_ENCODER]
        return model_initializer_fn(experiment_state)
