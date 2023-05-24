"""
clevr: encoder.py | Author : Sam Acquaviva.

Wrapper around the original list encoder from DreamCoder.
"""

from src.experiment_iterator import *
from src.models.model_loaders import (
    ModelLoaderRegistries,
    EXAMPLES_ENCODER,
    ModelLoader,
)
from src.task_loaders import TRAIN, TEST, ALL

from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.utilities import flatten
# from dreamcoder.domains.list.main import LearnedFeatureExtractor

ExamplesEncoderRegistry = ModelLoaderRegistries[EXAMPLES_ENCODER]

class LearnedFeatureExtractor(RecurrentFeatureExtractor):
    """The same as the LearnedFeatureExtractor from the list domain's main.py file, but maps the lexicon to strings."""
    H = 64
    
    special = None

    def tokenize(self, examples):
        def sanitize(l): return [z if z in self.lexicon else "?"
                                 for z_ in l
                                 for z in (z_ if isinstance(z_, list) else [z_])]

        tokenized = []
        for xs, y in examples:
            if isinstance(y, list):
                y = ["LIST_START"] + y + ["LIST_END"]
            else:
                y = [y]
            y = sanitize(y)
            if len(y) > self.maximumLength:
                return None

            serializedInputs = []
            for xi, x in enumerate(xs):
                if isinstance(x, list):
                    x = ["LIST_START"] + x + ["LIST_END"]
                else:
                    x = [x]
                x = sanitize(x)
                if len(x) > self.maximumLength:
                    return None
                serializedInputs.append(x)

            tokenized.append((tuple(serializedInputs), y))

        return tokenized

    def __init__(self, tasks, testingTasks=[], cuda=False):
        self.lexicon = set(flatten((t.examples for t in tasks + testingTasks), abort=lambda x: isinstance(
            x, str))).union({"LIST_START", "LIST_END", "?"})
        self.lexicon = list(map(str, self.lexicon))     # Convert all vocabulary to strings.

        # Calculate the maximum length
        self.maximumLength = float('inf') # Believe it or not this is actually important to have here
        self.maximumLength = max(len(l)
                                 for t in tasks + testingTasks
                                 for xs, y in self.tokenize(t.examples)
                                 for l in [y] + [x for x in xs])

        self.recomputeTasks = True

        super(
            LearnedFeatureExtractor,
            self).__init__(
            lexicon=list(
                self.lexicon),
            tasks=tasks,
            cuda=cuda,
            H=self.H,
            bidirectional=True)

@ExamplesEncoderRegistry.register
class ListFeatureExamplesEncoder(ModelLoader):
    """Loads the List Feature Extractor class. Note that this does not return an initialized model. 
    It returns the class that can be instantiated from the experiment state, with other params set. 
    Original source: dreamcoder/domains/list/main.py"""

    name = "list"

    def load_model_initializer(self, experiment_state, **kwargs):
        def experiment_state_initializer(exp_state):
            all_train_tasks = exp_state.get_tasks_for_ids(
                task_split=TRAIN, task_ids=ALL
            )
            all_test_tasks = exp_state.get_tasks_for_ids(task_split=TEST, task_ids=ALL)
            return LearnedFeatureExtractor(
                tasks=all_train_tasks,
                testingTasks=all_test_tasks,
                **kwargs
            )

        return experiment_state_initializer
