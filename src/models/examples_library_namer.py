"""
examples_library_namer.py | Author : Catherine Wong.

Example-driven library namer that prompts Codex to produce names for functions given examples of 
their behavior.

"""
from dbm.ndbm import library
from tabnanny import verbose
import numpy as np
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.laps_grammar import LAPSGrammar
import src.models.model_loaders as model_loaders
from src.task_loaders import ALL, TRAIN, LANGUAGE, PROGRAMS
from src.models.codex_base import *


from dreamcoder.type import *
from data.drawings.drawings_primitives import tfloat

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_NAMER]

DEFAULT_HEADER = ""


@ModelRegistry.register
class CodexExamplesLibraryNamer(CodexBase, model_loaders.ModelLoader):
    name = "codex_examples_library_namer"
    LIBRARY_DEFAULT_SEPARATOR = "\n"

    query_results_file = "codex_library_namer_results.json"

    # How to select names.
    TOP_1 = "top_1"
    SAMPLE_LOG_PROBS = "sample_log_probs"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexExamplesLibraryNamer(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def generate_library_names(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        n_samples_per_prompt: int = 100,
        n_train_library_functions_per_prompt: int = 10,
        n_train_tasks_per_library_function: int = 5,
        temperature: float = 0.1,
        max_tokens: int = 256,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        language_separator: str = CodexBase.DEFAULT_LANGUAGE_SEPARATOR,
        engine: str = CodexBase.DEFAULT_ENGINE,
        debug: bool = False,
        verbose_prompt: bool = True,
        function_name_class: list = LAPSGrammar.HUMAN_READABLE,
        mask_function_class=LAPSGrammar.NUMERIC_FUNCTION_NAMES,
        prompt_example_types: list = [LANGUAGE, PROGRAMS],
        name_selection_criteria: str = TOP_1,
        print_every_prompt_idx=1,
    ):
        """
        Queries Codex API to generate new names for library functions.

        n_train_library_functions_per_prompt : how many named library functions to include as examples.

        n_train_tasks_per_library_function : how many training tasks to include per library function.
        """
        query_results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            self.query_results_file,
        )

        # Get library functions for prompt and to be named.
        train_library_functions = self._get_train_library_functions(
            experiment_state, n_train_library_functions_per_prompt, function_name_class,
        )
        unnamed_invention_functions = self._get_unnamed_inventions(
            experiment_state, function_name_class
        )

        # Build example usages for train and invention library functions.
        train_example_usages = [
            self._build_usage_example(
                library_function,
                experiment_state,
                task_ids_in_splits,
                function_name_class,
                n_train_tasks_per_library_function=n_train_tasks_per_library_function,
                prompt_example_types=prompt_example_types,
                use_base_dsl=True,
            )
            for library_function in train_library_functions
        ]
        invention_example_usages = [
            self._build_usage_example(
                library_function,
                experiment_state,
                task_ids_in_splits,
                function_name_class,
                n_train_tasks_per_library_function=n_train_tasks_per_library_function,
                prompt_example_types=prompt_example_types,
                use_base_dsl=False,
            )
            for library_function in unnamed_invention_functions
        ]
        train_example_usages = [
            self._build_masked_usage_example(
                usage, experiment_state, [function_name_class], [mask_function_class]
            )
            for usage in train_example_usages
        ]

        # Iteratively construct prompts and cache names.
        for invention_example_usage in invention_example_usages:
            invention_example_usage = self._build_masked_usage_example(
                invention_example_usage,
                experiment_state,
                [function_name_class, mask_function_class],  # Mask any inventions also
                [mask_function_class],
            )
            prompt = LibraryNamePrompt(
                experiment_state,
                body_example_usages=train_example_usages,
                final_example_usage=invention_example_usage,
            )
            # if verbose_prompt:
            #     print(str(prompt))
            completion, cache_used = self.get_completion_for_prompt(
                experiment_state=experiment_state,
                prompt_text=str(prompt),
                query_results_filepath=query_results_filepath,
                n_samples_per_prompt=n_samples_per_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=separator,
                use_cached=False,
                debug=debug,
                logprobs=1,
            )
            if completion is not None:
                alternate_names = [
                    (choice["text"], np.mean(choice["logprobs"]["token_logprobs"]))
                    for choice in completion["choices"]
                ]
                alternate_name, log_prob = self._select_name(
                    alternate_names, name_selection_criteria
                )
                alternate_name = "_".join(alternate_name.split())
                alternate_name = experiment_state.models[
                    model_loaders.GRAMMAR
                ].set_function_name(
                    str(invention_example_usage[self.LIBRARY_FUNCTION]),
                    name_class=function_name_class,
                    name=alternate_name,
                )
                if verbose_prompt:
                    print(alternate_names)
                    print(
                        f"Setting function name for {invention_example_usage[self.LIBRARY_FUNCTION]} to {alternate_name} w/ log_p = {log_prob}"
                    )
                #  TODO[CW]: cache this out for results analysis

    def _select_name(self, alternate_names, name_selection_criteria):
        alternate_names = sorted(alternate_names, key=lambda c: c[-1], reverse=True)
        if name_selection_criteria == self.TOP_1:
            return alternate_names[0]
        elif name_selection_criteria == self.SAMPLE_LOG_PROBS:
            # Sample according to probability.
            names, probabilities = zip(*alternate_names)
            return np.random.choice(alternate_names, p=probabilities)[0]
        else:
            assert False

    def _new_library_usages(self):
        usages = {
            self.LIBRARY_FUNCTION: None,
            self.EXAMPLES: None,
        }
        return usages

    def _new_usage_example(self):
        usage_example = {
            self.TASK_ID: None,
            LANGUAGE: None,
            PROGRAMS: None,
            self.MASKED_NAMED_PROGRAMS: None,
        }
        return usage_example

    def _get_train_library_functions(
        self,
        experiment_state,
        n_train_library_functions_per_prompt,
        function_name_class,
        exclude_types=[
            tint,
            tfloat,
            arrow(tfloat, tfloat, tfloat),
            arrow(tfloat, tfloat),
        ],  # Excludes operators over numbers.
    ):
        """Gets n_training library functions from the base DSL
         that already have names for function_name_class."""
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        train_library_functions = [
            p
            for p in grammar.primitives
            if not p.isInvented
            and p.tp not in exclude_types
            and function_name_class in grammar.function_names[str(p)]
        ]
        num_to_sample = min(
            len(train_library_functions), n_train_library_functions_per_prompt
        )

        return rng.choice(train_library_functions, num_to_sample, replace=False,)

    def _get_unnamed_inventions(self, experiment_state, function_name_class):
        """Gets inventions in the grammar that do not have names for function_name_class."""
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        unnamed_library_functions = [
            p
            for p in grammar.primitives
            if p.isInvented
            and function_name_class not in grammar.function_names[str(p)]
        ]
        return unnamed_library_functions

    def _get_library_function_usages(self, library_function, programs):
        usage_programs = []
        for p in programs:
            if str(library_function) in p.left_order_tokens():
                usage_programs.append(p)

        return usage_programs

    def _build_usage_example(
        self,
        library_function,
        experiment_state,
        task_ids_in_splits,
        function_name_class,
        n_train_tasks_per_library_function,
        prompt_example_types,
        use_base_dsl,
    ):
        """
        Builds a 'usage' object containing examples of usage and the name for function_name_class if it exists.
        """
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        candidate_usage_examples = []
        for task_id in task_ids_in_splits[TRAIN]:
            frontier = experiment_state.get_frontiers_for_ids(TRAIN, [task_id])[0]
            programs_for_task = [e.program for e in frontier.entries]
            if use_base_dsl:
                programs_for_task = [e.betaNormalForm() for e in programs_for_task]
            language_for_task = experiment_state.get_language_for_ids(TRAIN, [task_id])[
                0
            ]
            program_usages = self._get_library_function_usages(
                library_function, programs_for_task
            )

            if len(program_usages) > 0:
                usage_example = self._new_usage_example()
                usage_example[self.TASK_ID] = task_id
                if LANGUAGE in prompt_example_types:
                    usage_example[LANGUAGE] = rng.choice(language_for_task)
                if PROGRAMS in prompt_example_types:
                    usage_example[PROGRAMS] = rng.choice(programs_for_task)
                candidate_usage_examples.append(usage_example)
        # Select usage examples.
        num_to_sample = min(
            len(candidate_usage_examples), n_train_tasks_per_library_function
        )
        usage_examples = rng.choice(
            candidate_usage_examples, num_to_sample, replace=False
        )
        usages = self._new_library_usages()
        usages[self.LIBRARY_FUNCTION] = library_function
        usages[self.EXAMPLES] = usage_examples
        return usages

    def _build_masked_usage_example(
        self,
        usages,
        experiment_state,
        function_name_classes=[LAPSGrammar.HUMAN_READABLE],
        mask_function_classes=[LAPSGrammar.NUMERIC_FUNCTION_NAMES],
    ):
        """Generates a masked_named_program example where all other functions are named using function_name_class
        and a given function is named with mask_function_class
        """
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        for example in usages[self.EXAMPLES]:
            if example[PROGRAMS] is not None:
                original_program = example[PROGRAMS]
                example[self.MASKED_NAMED_PROGRAMS] = grammar.show_program(
                    original_program,
                    name_classes=function_name_classes,
                    mask_primitives=[usages[self.LIBRARY_FUNCTION]],
                    mask_name_classes=mask_function_classes,
                )
        return usages

