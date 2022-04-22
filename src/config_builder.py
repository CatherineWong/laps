"""
task_loaders.py | Author : Gabe Grand
Utilities for autogenerating configs for experiments based on templates.

"""

import json
import os
from re import L

from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import (
    LIBRARY_LEARNER,
    PROGRAM_REWRITER,
    SAMPLE_GENERATOR,
    INITIALIZE_GROUND_TRUTH,
)
from src.task_loaders import (
    ALL,
    DEFAULT,
    GroundTruthOrderedTaskBatcher,
    LANGUAGE,
    SYNTHETIC,
    HUMAN,
)
import data.drawings.make_tasks as drawing_tasks

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
    "n_samples": 100,
    "n_train_programs_per_prompt": 25,
    "temperature": 0.75,
    "max_tokens": 256,
    "function_name_classes": ["default"],
}
DEFAULT_LANGUAGE = {
    "logo": "synthetic",
    "re2": "synthetic",
    "clevr": "synthetic",
    "drawings": "human",
    "drawings_nuts_bolts": "human",
    "drawings_wheels": "human",
    "drawings_furniture": "human",
    "drawings_dials": "human",
}


def get_domain_metadata(domain: str, language: str):
    if language == DEFAULT:
        language = DEFAULT_LANGUAGE[domain]

    METADATA = {
        "logo": {
            "tasks_loader": "compositional_graphics_200",
            "task_language_loader": f"compositional_graphics_200_{language}",
            "ocaml_special_handler": "LOGO",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200],
        },
        "clevr": {
            "tasks_loader": "clevr",
            "task_language_loader": f"clevr_{language}",
            "ocaml_special_handler": "clevr",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 191],
        },
        "re2": {
            "tasks_loader": "re2",
            "task_language_loader": f"re2_{language}",
            "ocaml_special_handler": "re2",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200, 300, 400, 491],
        },
    }

    # Metadata for each drawing task domain
    METADATA["drawings"] = {
        "tasks_loader": "drawings",
        "task_language_loader": f"drawings_{language}",
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
            "task_language_loader": f"drawings_{language}_{drawing_domain}",
            "ocaml_special_handler": "drawings",
            "global_batch_sizes": [5, 10, 15, 25, 50, 50, 100, 150, 200],
        }
        METADATA[drawing_domain_name] = drawing_domain_metadata

    return METADATA[domain]


def build_config(
    experiment_type: str,
    domain: str,
    language: str,
    experiment_id: str = None,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
    max_iterations: int = 1,
    task_batcher: str = GroundTruthOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    codex_params: dict = DEFAULT_CODEX_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
):
    config = {}
    config.update(
        build_config_body(
            experiment_type=experiment_type,
            language=language,
            domain=domain,
            max_iterations=max_iterations,
            task_batcher=task_batcher,
            global_batch_size=global_batch_size,
            stitch_params=stitch_params,
            codex_params=codex_params,
            compute_likelihoods=compute_likelihoods,
            compute_description_lengths=compute_description_lengths,
        )
    )
    config.update(
        build_config_metadata(
            domain=domain,
            language=language,
            experiment_type=experiment_type,
            experiment_id=experiment_id,
            global_batch_size=global_batch_size,
            output_directory=output_directory,
            random_seed=random_seed,
        )
    )
    return config


def build_config_metadata(
    domain: str,
    experiment_type: str,
    experiment_id: str = None,
    language: str = DEFAULT,
    global_batch_size: int = ALL,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
):
    domain_meta = get_domain_metadata(domain, language)

    if experiment_id is None:
        experiment_id = experiment_type
    if LANGUAGE in experiment_id:
        if LANGUAGE == DEFAULT:
            language = DEFAULT_LANGUAGE[domain]
        experiment_id += f"_{language}"

    export_directory = os.path.join(
        output_directory,
        "outputs",
        "domains",
        domain,
        experiment_id,
        f"seed_{random_seed}",
    )
    log_directory = os.path.join(
        output_directory,
        "logs",
        "domains",
        domain,
        experiment_id,
        f"seed_{random_seed}",
    )

    experiment_id = f"{experiment_id}_{global_batch_size}"

    return {
        "metadata": {
            "experiment_id": experiment_id,
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
    language: str,
    domain: str,
    max_iterations: int = 1,
    task_batcher: str = GroundTruthOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    codex_params: dict = DEFAULT_CODEX_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
):
    template_path = os.path.join(
        DEFAULT_TEMPLATE_DIR, f"template_{experiment_type}.json"
    )
    with open(template_path, "r") as f:
        config = json.load(f)

    domain_meta = get_domain_metadata(domain, language)

    model_initializers = config["model_initializers"]
    model_initializers[0]["model_loader"] = domain_meta["ocaml_special_handler"]
    config["model_initializers"] = model_initializers

    config["experiment_iterator"]["max_iterations"] = max_iterations
    config["experiment_iterator"]["task_batcher"]["model_type"] = task_batcher
    config["experiment_iterator"]["task_batcher"]["params"][
        "global_batch_size"
    ] = global_batch_size

    # Use defaults for any unspeficied parameters
    _codex_params = DEFAULT_CODEX_PARAMS
    _codex_params.update(codex_params)
    _stitch_params = DEFAULT_STITCH_PARAMS
    _stitch_params.update(stitch_params)

    loop_blocks = []
    for block in config["experiment_iterator"]["loop_blocks"]:
        if block.get("model_type") == SAMPLE_GENERATOR:
            block["params"].update(_codex_params)
        if block.get("model_type") == LIBRARY_LEARNER:
            block["params"].update(_stitch_params)
        if (
            block.get("model_type")
            in [LAPSGrammar.GRAMMAR, SAMPLE_GENERATOR, PROGRAM_REWRITER,]
            or block.get("state_fn") == INITIALIZE_GROUND_TRUTH
        ):
            block["params"].update(
                {"compute_likelihoods": compute_likelihoods,}
            )
        loop_blocks.append(block)
    config["experiment_iterator"]["loop_blocks"] = loop_blocks

    return config
