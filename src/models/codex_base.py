"""
codex_base.py | Author: Catherine Wong, Gabe Grand.

Base class containing utilities for working with the Codex language model.
"""

import os
import time

import openai
import json
from openai.error import InvalidRequestError, RateLimitError
from openai.api_resources.completion import Completion

from src.models.laps_grammar import LAPSGrammar
import src.models.model_loaders as model_loaders
from src.task_loaders import LANGUAGE


MAX_CODEX_TOKENS = 4096

DEFAULT_LINE_SEPARATOR = "\n"


class CodexBase(object):
    DEFAULT_ENGINE = "davinci-codex"
    DEFAULT_SEPARATOR = "\n"
    DEFAULT_LANGUAGE_SEPARATOR = "# "

    LIBRARY_FUNCTION = "library_function"
    EXAMPLES = "examples"
    MASKED_NAMED_PROGRAMS = "masked_named_programs"
    TASK_ID = "task_id"

    def __init__(self, experiment_state=None):
        super().__init__()
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is not set. Please set this in the shell via `export OPENAI_API_KEY=...`"
            )
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_completion_for_prompt(
        self,
        experiment_state,
        prompt_text,
        query_results_filepath,
        n_samples_per_prompt,
        temperature,
        max_tokens,
        engine,
        separator,
        use_cached,
        debug,
        logprobs,
    ):
        if debug:
            # Debugging query that returns programs.
            cache_used = True
            completion = self.query_mock(
                experiment_state, n_samples=n_samples_per_prompt
            )
        elif use_cached and os.path.exists(query_results_filepath):
            cache_used = True
            print("Using cached examples....")
            # For debugging only - does not verify that the cached completion matches the desired query parameters
            with open(query_results_filepath, "r") as f:
                completion_data = json.load(f)["completion"]
            completion = Completion()
            completion.refresh_from(completion_data)
        else:
            cache_used = False
            completion = self.query_codex(
                prompt_text,
                n_samples=n_samples_per_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine,
                separator=separator,
                logprobs=logprobs,
            )
        return completion, cache_used

    def query_codex(
        self,
        prompt: str,
        n_samples: int,
        temperature: float = 0.75,
        max_tokens: int = 256,
        engine: str = DEFAULT_ENGINE,
        separator: str = DEFAULT_SEPARATOR,
        top_p=None,
        logprobs=None,
        max_attempts_rate_limit=5,
        rate_limit_seconds=60,
    ):
        pause_for_rate_limit = False
        completion = None
        for idx in range(max_attempts_rate_limit):
            if pause_for_rate_limit:
                print(
                    f"ERR: Codex rate limit. On attempt {idx}/{max_attempts_rate_limit} after waiting {rate_limit_seconds}s."
                )
                time.sleep(rate_limit_seconds)
                rate_limit_seconds *= 2  # Exponential backoff
            try:
                completion = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature if top_p is None else 1.0,
                    top_p=top_p if temperature is None else 1.0,
                    n=n_samples,
                    stop=separator,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                )
                return completion
            except RateLimitError as e:
                print(e)
                pause_for_rate_limit = True
                completion = None
            except Exception as e:
                print(e)
                completion = None
                return completion

        return completion


class LibraryNamePrompt(object):

    DEFAULT_PREFIX_EXAMPLE = "/* Example usages of {} */"
    DEFAULT_PREFIX_PROGRAM = ""
    DEFAULT_PREFIX_LANGUAGE = "# Human readable description: "
    DEFAULT_PREFIX_READABLE_NAME = "# Can you guess what {} should be named : "

    def __init__(
        self,
        experiment_state,
        body_example_usages,
        final_example_usage,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        function_name_class=LAPSGrammar.HUMAN_READABLE,
        mask_function_class=LAPSGrammar.NUMERIC_FUNCTION_NAMES,
    ):
        self.body_example_usages = body_example_usages
        self.final_example_usage = final_example_usage
        self.line_separator = line_separator

        self.function_name_class = function_name_class
        self.mask_function_class = mask_function_class

        self.grammar = experiment_state.models[model_loaders.GRAMMAR]

    def example_usage_to_string(self, example_usages, is_final_line=False):
        example_lines = []

        original_function_name = str(example_usages[CodexBase.LIBRARY_FUNCTION])
        masked_function_name = self.grammar.function_names[original_function_name][
            self.mask_function_class
        ]

        example_lines.append(self.DEFAULT_PREFIX_EXAMPLE.format(masked_function_name))
        for usage in example_usages[CodexBase.EXAMPLES]:
            if usage[LANGUAGE] is not None:
                example_lines.append(self.DEFAULT_PREFIX_LANGUAGE + usage[LANGUAGE])
            if usage[CodexBase.MASKED_NAMED_PROGRAMS] is not None:
                example_lines.append(
                    self.DEFAULT_PREFIX_PROGRAM + usage[CodexBase.MASKED_NAMED_PROGRAMS]
                )
        if not is_final_line:
            readable_function_name = self.grammar.function_names[
                original_function_name
            ][self.function_name_class]
            example_lines.append(
                self.DEFAULT_PREFIX_READABLE_NAME.format(masked_function_name)
                + readable_function_name
            )
        else:
            example_lines.append(
                self.DEFAULT_PREFIX_READABLE_NAME.format(masked_function_name)
            )

        return self.line_separator.join(example_lines)

    def __str__(self):
        prompt_text = ""

        # Convert the body examples into a prompt.
        for example_usage in self.body_example_usages:
            if len(example_usage[CodexBase.EXAMPLES]) < 1:
                continue
            prompt_text += self.example_usage_to_string(example_usage)
            prompt_text += self.line_separator
            prompt_text += self.line_separator
        # Convert the library function into a prompt, naming it using
        # the current grammar.
        prompt_text += self.example_usage_to_string(
            self.final_example_usage, is_final_line=True
        )
        return prompt_text
