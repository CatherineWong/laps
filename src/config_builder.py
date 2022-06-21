"""
task_loaders.py | Author : Gabe Grand
Utilities for autogenerating configs for experiments based on templates.

"""

import json
import os
from enum import Enum

import data.drawings.make_tasks as drawing_tasks
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import (
    AMORTIZED_SYNTHESIS,
    INITIALIZE_GROUND_TRUTH,
    LIBRARY_LEARNER,
    PROGRAM_REWRITER,
    SAMPLE_GENERATOR,
)
from src.task_loaders import ALL, GroundTruthOrderedTaskBatcher

DEFAULT_EXPERIMENT_DIR = "experiments_iterative"
DEFAULT_TEMPLATE_DIR = os.path.join(DEFAULT_EXPERIMENT_DIR, "templates")


DEFAULT_STITCH_PARAMS = {
    "max_arity": 2,
    "iterations": 1,
    "candidates_per_iteration": 1,
}

DEFAULT_CODEX_PARAMS = {
    "debug": False,
    "use_cached": False,
    "n_samples": 2,
    "n_samples_per_query": 5,
    "n_tasks_per_prompt": 20,
    "temperature": 0.75,
    "max_tokens": 256,
    "function_name_classes": ["default_no_inline", "numeric"],
}


class ExperimentType(str, Enum):
    ORACLE = "oracle"
    ORACLE_TRAIN_TEST = "oracle_train_test"
    STITCH = "stitch"
    STITCH_CODEX = "stitch_codex"
    STITCH_CODEX_LANGUAGE = "stitch_codex_language"
    STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST = (
        "stitch_codex_language_origin_random_test"
    )


def get_domain_metadata(domain: str):
    METADATA = {
        "logo": {
            "tasks_loader": "compositional_graphics_200",
            "task_language_loader": "compositional_graphics_200_synthetic",
            "ocaml_special_handler": "LOGO",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200],
            "examples_encoder": "logo_cnn_examples_encoder",
        },
        "clevr": {
            "tasks_loader": "clevr",
            "task_language_loader": "clevr_synthetic",
            "ocaml_special_handler": "clevr",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 191],
            "examples_encoder": "clevr_rnn_examples_encoder",
            "enumeration_timeout": 1000,
            "recognition_train_steps": 10000,
        },
        "re2": {
            "tasks_loader": "re2",
            "task_language_loader": "re2_synthetic",
            "ocaml_special_handler": "re2",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200, 300, 400, 491],
            "examples_encoder": "re2_rnn_examples_encoder",
            "enumeration_timeout": 720,
            "recognition_train_steps": 10000,
        },
    }

    # Metadata for each drawing task domain
    METADATA["drawings"] = {
        "tasks_loader": "drawings",
        "task_language_loader": f"drawings_human",
        "ocaml_special_handler": "drawings",
        "global_batch_sizes": [
            5,
            10,
            15,
            25,
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
        ],
    }
    for drawing_domain in drawing_tasks.TASK_DOMAINS:
        drawing_domain_name = "drawings_" + drawing_domain
        drawing_domain_metadata = {
            "tasks_loader": drawing_domain_name,
            "task_language_loader": f"drawings_human_{drawing_domain}",
            "ocaml_special_handler": "drawings",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 150, 200],
        }
        METADATA[drawing_domain_name] = drawing_domain_metadata

    return METADATA[domain]


def build_config(
    experiment_name: str,
    experiment_type: str,
    domain: str,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
    iterations: int = 1,
    task_batcher: str = GroundTruthOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    codex_params: dict = DEFAULT_CODEX_PARAMS,
    synthesizer_params: dict = {},
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = False,
):
    config = {}
    config.update(
        build_config_body(
            experiment_type=experiment_type,
            domain=domain,
            iterations=iterations,
            task_batcher=task_batcher,
            global_batch_size=global_batch_size,
            stitch_params=stitch_params,
            codex_params=codex_params,
            synthesizer_params=synthesizer_params,
            compute_likelihoods=compute_likelihoods,
            compute_description_lengths=compute_description_lengths,
            increment_task_batcher=increment_task_batcher,
        )
    )
    config.update(
        build_config_metadata(
            experiment_name=experiment_name,
            domain=domain,
            experiment_type=experiment_type,
            global_batch_size=global_batch_size,
            output_directory=output_directory,
            random_seed=random_seed,
        )
    )
    return config


def build_config_metadata(
    experiment_name: str,
    domain: str,
    experiment_type: str,
    global_batch_size: int = ALL,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
):
    domain_meta = get_domain_metadata(domain)

    export_directory = os.path.join(
        output_directory,
        "outputs",
        experiment_name,
        "domains",
        domain,
        experiment_type,
        f"seed_{random_seed}",
    )
    log_directory = os.path.join(
        output_directory,
        "logs",
        experiment_name,
        "domains",
        domain,
        experiment_type,
        f"seed_{random_seed}",
    )

    return {
        "metadata": {
            "experiment_name": experiment_name,
            "experiment_id": f"{experiment_type}_{global_batch_size}",
            "human_readable": "Autogenerated iterative experiment.",
            "export_directory": export_directory,
            "log_directory": log_directory,
            "tasks_loader": domain_meta["tasks_loader"],
            "task_language_loader": domain_meta["task_language_loader"],
            "export_with_timestamp": False,
            "resume_checkpoint_directory": None,
            "init_frontiers_from_checkpoint": False,
            "ocaml_special_handler": domain_meta["ocaml_special_handler"],
            "global_batch_sizes": domain_meta["global_batch_sizes"],
            "random_seed": random_seed,
        }
    }


def build_config_body(
    experiment_type: str,
    domain: str,
    iterations: int = 1,
    task_batcher: str = GroundTruthOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    codex_params: dict = DEFAULT_CODEX_PARAMS,
    synthesizer_params: dict = {},
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = False,
):
    template_path = os.path.join(
        DEFAULT_TEMPLATE_DIR, f"template_{experiment_type}.json"
    )
    with open(template_path, "r") as f:
        config = json.load(f)

    domain_meta = get_domain_metadata(domain)

    model_initializers = config["model_initializers"]
    model_initializers[0]["model_loader"] = domain_meta["ocaml_special_handler"]
    config["model_initializers"] = model_initializers

    # Set the domain-specific examples_encoder.
    for model_initializer in config["model_initializers"]:
        if model_initializer["model_type"] == "examples_encoder":
            model_initializer["model_loader"] = domain_meta["examples_encoder"]

    config["experiment_iterator"]["max_iterations"] = iterations
    config["experiment_iterator"]["task_batcher"]["model_type"] = task_batcher
    config["experiment_iterator"]["task_batcher"]["params"][
        "global_batch_size"
    ] = global_batch_size

    config["experiment_iterator"]["task_batcher"]["params"][
        "increment_at_global_iteration"
    ] = increment_task_batcher

    # Use defaults for any unspeficied parameters
    _codex_params = DEFAULT_CODEX_PARAMS
    _codex_params.update(codex_params)
    _stitch_params = DEFAULT_STITCH_PARAMS
    _stitch_params.update(stitch_params)
    _synthesizer_params = {
        "enumeration_timeout": domain_meta.get("enumeration_timeout", 1000),
        "recognition_train_steps": domain_meta.get("recognition_train_steps", 10000),
    }
    _synthesizer_params.update(synthesizer_params)

    loop_blocks = []
    for block in config["experiment_iterator"]["loop_blocks"]:
        if block.get("model_type") == SAMPLE_GENERATOR:
            block["params"].update(_codex_params)
        if block.get("model_type") == LIBRARY_LEARNER:
            block["params"].update(_stitch_params)
        if block.get("model_type") == AMORTIZED_SYNTHESIS:
            block["params"].update(_synthesizer_params)
        if (
            block.get("model_type")
            in [
                LAPSGrammar.GRAMMAR,
                SAMPLE_GENERATOR,
                PROGRAM_REWRITER,
            ]
            or block.get("state_fn") == INITIALIZE_GROUND_TRUTH
        ):
            block["params"].update(
                {
                    "compute_likelihoods": compute_likelihoods,
                }
            )
        loop_blocks.append(block)
    config["experiment_iterator"]["loop_blocks"] = loop_blocks

    return config
