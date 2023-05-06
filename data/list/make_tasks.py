"""
re2: make_tasks.py | Author : Sam Acquaviva.

Loading tasks and language for the list domain.
"""
import os

from src.task_loaders import *
# from data.re2.grammar import *
import dreamcoder.domains.list.makeListTasks as makeListTasks
# from dreamcoder.domains.re2.re2Primitives import *
from dreamcoder.domains.list.main import retrieveJSONTasks, testTrainSplit, sortBootstrap, make_list_bootstrap_tasks
from dreamcoder.program import Program

DOMAIN_NAME = "list"

ROOT_DIR = os.getcwd()
DEFAULT_DATA_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")

DEFAULT_TRAIN_TEST_SPLIT = 0.8

DEFAULT_DATASET = "Lucas-depth3"
DEFAULT_TASKS_DIRECTORY = os.path.join(ROOT_DIR, f"dreamcoder/data/")
DEFAULT_LANGUAGE_DIRECTORY = os.path.join(ROOT_DIR, f"data/{DOMAIN_NAME}/language/")


@TaskLoaderRegistry.register
class Re2Loader(TaskDataLoader):
    name = DOMAIN_NAME

    def load_tasks(self):
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

        test_tasks, train_tasks = testTrainSplit(tasks, DEFAULT_TRAIN_TEST_SPLIT)

        tasks = {TRAIN: train_tasks, TEST: test_tasks}
        # for split in tasks.keys():
        #     for t in tasks[split]:
        #         t.supervisedSolution = Program.parse(t.name).betaNormalForm()
        #         t.groundTruthProgram = t.supervisedSolution

        return tasks

@TaskLanguageLoaderRegistry.register
class Re2SyntheticLanguageLoader(TaskDataLoader):
    name = "list_synthetic"

    def load_task_language(self):
        dataset_path = os.path.join(
            DEFAULT_LANGUAGE_DIRECTORY, DEFAULT_DATASET, SYNTHETIC
        )
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])
