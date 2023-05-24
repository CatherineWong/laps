"""
re2: make_tasks.py | Author : Sam Acquaviva.

Loading tasks and language for the list domain.
"""
import os

from src.task_loaders import *
from dreamcoder.domains.list.main import retrieveJSONTasks, testTrainSplit, sortBootstrap, make_list_bootstrap_tasks

DOMAIN_NAME = "list"

ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")

DEFAULT_TRAIN_TEST_SPLIT = 0.7

DEFAULT_DATASET = "Lucas-old"
DEFAULT_TASKS_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")
DEFAULT_LANGUAGE_DIRECTORY = os.path.join(ROOT_DIR, f"data/{DOMAIN_NAME}/language/")


def load_list_tasks(split_seed = 0):
    """Loads the list domain tasks. Largely taken from the list domain's main.py file."""

    old_list_tasks_path = os.path.join(DEFAULT_TASKS_DIRECTORY, "list_tasks.json")
    new_list_tasks_path = os.path.join(DEFAULT_TASKS_DIRECTORY, "list_tasks2.json")

    tasks = {
        "Lucas-old": lambda: retrieveJSONTasks(old_list_tasks_path) + sortBootstrap(),
        "bootstrap": make_list_bootstrap_tasks,
        "sorting": sortBootstrap,
        "Lucas-depth1": lambda: retrieveJSONTasks(new_list_tasks_path)[:105],
        "Lucas-depth2": lambda: retrieveJSONTasks(new_list_tasks_path)[:4928],
        "Lucas-depth3": lambda: retrieveJSONTasks(new_list_tasks_path),
    }[DEFAULT_DATASET]()

    test_tasks, train_tasks = testTrainSplit(tasks, DEFAULT_TRAIN_TEST_SPLIT, seed=split_seed)
    tasks = {TRAIN: train_tasks, TEST: test_tasks}
    # for split in tasks.keys():
    #     for t in tasks[split]:
    #         t.supervisedSolution = Program.parse(t.name).betaNormalForm()
    #         t.groundTruthProgram = t.supervisedSolution

    return tasks


@TaskLoaderRegistry.register
class Re2Loader(TaskDataLoader):
    name = DOMAIN_NAME

    def load_tasks(self):
        return load_list_tasks()


@TaskLanguageLoaderRegistry.register
class Re2SyntheticLanguageLoader(TaskDataLoader):
    name = "list_synthetic"

    def load_task_language(self):

        # Just use the language names as the dataset language. 
        # So, first load the tasks.
        tasks = load_list_tasks()

        # Then, parse the language names from the tasks into vocab and language.
        # Currently, just splitting on spaces.
        # TODO: Use a better vocab generator.
        # TODO: Store the vocab / language in a file rather than loading tasks + parsing.
        language, vocab = {}, {}
        for split, split_tasks in tasks.items():
            language[split] = {task.name: task.name for task in split_tasks}
            vocab[split] = list(set([token for task in split_tasks for token in task.name.split(" ")]))

        return language, vocab
