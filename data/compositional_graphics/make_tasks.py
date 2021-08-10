"""
compositional_graphics: make_tasks.py | Author: Catherine Wong.

Utility functions for loading tasks and language for the compositional graphics domain. This domain was originally collected and used in the LAPS-ICML 2021 paper and can be found in the dreamcoder/logo domain.
"""
import os, dill
from src.experiment_iterator import (
    TaskDataLoader,
    TaskLoaderRegistry,
    TaskLanguageLoaderRegistry,
    TASKS,
    LANGUAGE,
    TRAIN,
    TEST,
    HUMAN,
    SYNTHETIC,
)

DEFAULT_DATA_DIRECTORY = "data/compositional_graphics"


@TaskLoaderRegistry.register
class CompositionalGraphics200Loader(TaskDataLoader):
    name = "compositional_graphics_200"

    def load_tasks(self):
        dataset_path = os.path.join(DEFAULT_DATA_DIRECTORY, TASKS, self.name)
        tasks = {TRAIN: [], TEST: []}
        for split in tasks.keys():
            split_path = os.path.join(dataset_path, split)
            task_files = sorted(
                [
                    os.path.join(split_path, t)
                    for t in os.listdir(split_path)
                    if ".p" in t
                ],
                key=lambda p: os.path.basename(p),
            )
            for task_file in task_files:
                with open(task_file, "rb") as f:
                    t = dill.load(f)
                    t.nearest_name = None
                    tasks[split].append(t)
        return tasks


@TaskLanguageLoaderRegistry.register
class CompositionalGraphics200HumanLanguageLoader(TaskDataLoader):
    name = "compositional_graphics_200_human"

    def load_task_language(self):
        task_dataset = CompositionalGraphics200Loader.name
        dataset_path = os.path.join(
            DEFAULT_DATA_DIRECTORY, LANGUAGE, task_dataset, HUMAN
        )
        return self.load_task_language_for_directory(
            dataset_path, splits=[TRAIN, TEST]
        )


@TaskLanguageLoaderRegistry.register
class CompositionalGraphics200SyntheticLanguageLoader(TaskDataLoader):
    name = "compositional_graphics_200_synthetic"

    def load_task_language(self):
        task_dataset = CompositionalGraphics200Loader.name
        dataset_path = os.path.join(
            DEFAULT_DATA_DIRECTORY, LANGUAGE, task_dataset, HUMAN
        )
        return self.load_task_language_for_directory(
            dataset_path, splits=[TRAIN, TEST]
        )
