"""
laps_language_encoder.py | Author : Catherine Wong.

Utility wrapper function to load the LAPS language encoder.
"""

from src.task_loaders import *
import src.models.model_loaders as model_loaders

from dreamcoder.parser import TokenRecurrentFeatureExtractor

LanguageEncoderModelRegistry = model_loaders.ModelLoaderRegistries[
    model_loaders.LANGUAGE_ENCODER
]


@LanguageEncoderModelRegistry.register
class LapsRNNLanguageEncoder(model_loaders.ModelLoader):
    """Loads the LAPS language RNN encoder. Original source: dreamcoder/parser.py"""

    name = "laps_rnn_language_encoder"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)

            train_language_data = exp_state.task_language[TRAIN]
            test_language_data = exp_state.task_language[TEST]
            all_language_data = {**train_language_data, **test_language_data}

            train_vocabulary = list(exp_state.task_vocab[TRAIN])

            # Make the local directory.
            return TokenRecurrentFeatureExtractor(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                language_data=all_language_data,
                cuda=False,
                lexicon=train_vocabulary,
                **kwargs,
            )

        return experiment_state_initializer

