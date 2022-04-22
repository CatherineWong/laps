"""
examples_library_namer.py | Author : Catherine Wong.

Example-driven library namer that prompts Codex to produce names for functions given examples of 
their behavior.

"""
from dbm.ndbm import library
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

    LIBRARY_FUNCTION = ("library_function",)
    EXAMPLES = "examples"
    MASKED_NAMED_PROGRAMS = "masked_named_programs"
    TASK_ID = ("task_id",)

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return CodexExamplesLibraryNamer(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

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
    
    

    def generate_library_names(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: list,
        n_train_library_functions_per_prompt: int = 5,
        n_train_tasks_per_library_function: int = 5,
        temperature: float = 0.75,
        max_tokens: int = 256,
        separator: str = CodexBase.DEFAULT_SEPARATOR,
        language_separator: str = CodexBase.DEFAULT_LANGUAGE_SEPARATOR,
        engine: str = CodexBase.DEFAULT_ENGINE,
        debug: bool = False,
        verbose_prompt: bool = False,
        function_name_class: list = LAPSGrammar.HUMAN_READABLE,
        prompt_example_types: list = [LANGUAGE, PROGRAMS],
        print_every_prompt_idx=1,
    ):
        """
        Queries Codex API to generate new names for library functions.

        n_train_library_functions_per_prompt : how many named library functions to include as examples.

        n_train_tasks_per_library_function : how many training tasks to include per library function.
        """
        train_library_functions = self._get_train_library_functions(
            experiment_state, n_train_library_functions_per_prompt, function_name_class,
        )
        unnamed_invention_functions = self._get_unnamed_inventions(
            experiment_state, function_name_class
        )
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

        # Construct masked examples for training prompt.
        train_example_usages = [
            self.
            for usage in train_example_usages
        ]

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

    def _build_fixed_prompt_header(
        self,
        experiment_state,
        prompt_comment_header,
        prompt_with_base_dsl,
        skip_types=["int"],
    ):
        prompt_header = ""
        if prompt_comment_header is not None:
            prompt_header += prompt_comment_header
        if prompt_with_base_dsl:
            prompt_header += CodexLibraryNamer.DEFAULT_BASE_DSL_HEADER
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            for p in grammar.primitives:
                if p.isInvented:
                    # TODO: add previously named inventions here.
                    continue
                if str(p.tp) in skip_types:
                    continue
                prompt = "# Original function name: \n"  # TODO: use the numeric input name here.
                prompt += f"{p}\n"  # TODO: use the human readable names; give examples of usage - match the form.
                prompt += f"# Functionality: {p.function_comment}\n"
                prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
                prompt += (
                    f"{p.alternate_names[-1]}"  # TODO: use the human readable name.
                )
                prompt += "\n\n"
                prompt_header += prompt

        return prompt_header

    def _build_invention_prompt(
        self,
        experiment_state,
        invention,
        prompt_with_task_language,
        prompt_with_n_example_programs,
        body_name_class,
        input_name_classes,
        output_name_class,
    ):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        prompt = ""
        prompt += CodexLibraryNamer.DEFAULT_INVENTION_HEADER

        input_function_name = grammar.get_name(str(invention), input_name_classes)
        prompt += "# Original function name: \n"
        prompt += input_function_name + "\n"

        if prompt_with_n_example_programs > 0:
            example_usages = self._get_example_usages(
                experiment_state, invention, prompt_with_n_example_programs
            )
            prompt += (
                f"# Here are {prompt_with_n_example_programs} examples of its usage: "
                + "\n"
            )
            # TODO: add language; more intelligent example usage selection.
            example_programs = [
                grammar.show_program(
                    example,
                    name_classes=[body_name_class] + input_name_classes,
                    debug=True,
                )
                for example in example_usages.values()
            ]
            prompt += "\n".join(example_programs) + "\n"
        prompt += "# Function body: \n"
        function_body = str(
            grammar.show_program(
                invention.betaNormalForm(), name_classes=[body_name_class]
            )
        )
        prompt += function_body + "\n"
        prompt += f"# Give an alternate verbose, human-readable name for this function that describes what it does. Prefix it with {grammar.function_prefix}_ \n"
        prompt += f"{grammar.function_prefix}_"

        return prompt

    def _get_example_usages(self, experiment_state, primitive, n_examples):
        """
        :ret: [(task, example) for n_examples using the primitive]
        """
        # TODO: find examples where its not used along with inventions.
        example_usages = dict()
        for task, frontier in experiment_state.task_frontiers[TRAIN].items():
            for e in frontier.entries:
                if str(primitive) in e.tokens and not task in example_usages:
                    example_usages[task] = e.program
                    if len(example_usages) == n_examples:
                        return example_usages
        return example_usages

    def _get_inventions_to_name(
        self, experiment_state, inventions_to_name, output_name_class
    ):
        """
        :ret: [array of Invention expressions to name]
        """
        # Get inventions.
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        inventions = [p for p in grammar.primitives if p.isInvented]
        if inventions_to_name == ALL:
            pass
        elif inventions_to_name == self.ALL_UNNAMED:
            inventions = [
                i
                for i in inventions
                if not grammar.has_alternate_name(i, output_name_class)
            ]
        inventions = sorted(inventions, key=lambda p: str(p))
        return inventions

