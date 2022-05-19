"""
re2: encoder.py | Author : Catherine Wong.

Utility function for loading the examples encoder for the re2 domain.
"""
from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL


from dreamcoder.domains.re2.main import StringFeatureExtractor

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]


@ExamplesEncoderRegistry.register
class Re2RNNExamplesEncoder(ModelLoader):
    """Loads the RE2 RNN encoder class. Note that this does not return an initialized model. It returns the class that can be instantiated from the experiment state, with other params set. Original source: dreamcoder/domains/re2/main.py"""

    name = "re2_rnn_examples_encoder"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)

            return StringFeatureExtractor(
                tasks=all_train_tasks, testingTasks=all_test_tasks, **kwargs
            )

        return experiment_state_initializer
