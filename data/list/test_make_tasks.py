"""
list: test_make_tasks | Author : Sam Acquaviva.
"""

from src.task_loaders import *
import data.list.make_tasks as to_test


def test_load_list_tasks():
    task_loader = TaskLoaderRegistry[to_test.Re2Loader.name]
    tasks = task_loader.load_tasks()
    for split in tasks:
        print(split, len(tasks[split]))
