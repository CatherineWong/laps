"""
visualize_iterative_experiment_drawings | Author: Catherine Wong.

Visualizes the results of the iterative experiments for the drawings domain.
Generates a visualization PDF at the top-level replication directory for this experiment.

Currently only visualizes

Assumes that the directory structure is:
    {experiments_directory}/{experiment_type}/{replication}/{experiment_type}_{batch_size}

Usage:
python visualize_iterative_experiment.py
    --experiments_directory experiments_iterative/outputs/domains
    --domain drawings_nuts_bolts
    --experiment_type stitch_codex_language_human
    --visualize_codex_results
    --visualize_inventions # Not yet implemented.
"""
import argparse
import itertools
import json
import os
import random
from dreamcoder.program import Program

from matplotlib import pyplot as plt
import numpy as np
import PIL

from src.models.sample_generator import CodexSampleGenerator
from data.drawings.grammar import DrawingGrammar
import data.drawings.drawings_primitives as drawings_primitives

DEFAULT_EXPERIMENTS_DIRECTORY = "experiments_iterative/outputs/domains"
DEFAULT_DRAWINGS_GRAMMAR = DrawingGrammar.new_uniform()
DEFAULT_IMAGES_PER_ROW = 10
CODEX_SAMPLE_VISUALIZATION = "codex_sample_visualization.png"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--experiments_directory", default=DEFAULT_EXPERIMENTS_DIRECTORY,
)
parser.add_argument(
    "--experiment_type", required=True, help="[stitch, stitch_codex, oracle]"
)
parser.add_argument(
    "--domain", required=True, help="[drawings_nuts_bolts, drawings_wheels...]"
)

parser.add_argument(
    "--visualize_codex_results",
    action="store_true",
    help="Visualize programs generated by Codex.",
)
parser.add_argument(
    "--visualize_inventions",
    action="store_true",
    help="Visualize inventions generated by Stitch.",
)


def get_replication_directories(args):
    experiment_top_level_directory = os.path.join(
        args.experiments_directory, args.domain, args.experiment_type
    )
    replication_directories = [
        os.path.join(experiment_top_level_directory, d)
        for d in os.listdir(experiment_top_level_directory)
        if "." not in d
    ]
    print(f"Visualizing experiments in {experiment_top_level_directory}.")
    print(f"Found n={len(replication_directories)} replications.")
    replication_directories = [
        (
            replication_directory,
            [d for d in os.listdir(replication_directory) if "." not in d],
        )
        for replication_directory in replication_directories
    ]
    return replication_directories


def parse_string_drawings_program(string_program):
    string_program = DEFAULT_DRAWINGS_GRAMMAR.show_program(
        string_program,
        input_name_class=[
            DrawingGrammar.HUMAN_READABLE
        ],  # TODO: move this into the codex_results.
    )
    p = Program.parse(string_program)
    return p


def get_sampled_programs(codex_results):
    # Gets sampled programs out of the codex results.
    return [
        parse_string_drawings_program(string_program)
        for string_program in codex_results["programs_valid"]
    ]


def get_prompted_programs(codex_results, max_prompt_programs=16):
    prompt_examples = codex_results["prompt_example_types"]
    if "programs" not in prompt_examples:
        return []

    if len(prompt_examples) > 1:
        p_idx = prompt_examples.index("programs")
        prompt_programs = list(itertools.chain(*codex_results["prompt_programs"]))
        # Exclude the final prompt example for now.
        prompt_programs = [
            p[p_idx] for p in prompt_programs if len(p) >= len(prompt_examples)
        ]

    else:
        prompt_programs = list(itertools.chain(*codex_results["prompt_programs"]))
    # Randomly sample per prompt.
    prompt_programs = random.sample(
        prompt_programs, min(max_prompt_programs, len(prompt_programs))
    )
    return [
        parse_string_drawings_program(string_program)
        for string_program in prompt_programs
    ]


def stack_images(images):
    min_img_shape = sorted([(np.sum(i.size), i.size) for i in images])[0][1]
    img_merge = np.vstack(
        (np.asarray(i.resize(min_img_shape, PIL.Image.ANTIALIAS)) for i in images)
    )
    img_merge = PIL.Image.fromarray(img_merge)
    return img_merge


def get_blank_spacer(images, percentage=1):
    # Adds a spacer that is percentage * height
    last_image = images[-1]

    spacer = PIL.Image.new(
        "RGB", int(last_image.size[0]), last_image.size[-1], (255, 255, 255),
    )
    return spacer


def visualize_codex_results(args, replication_directory, replication_subdirectories):
    codex_results_figure = []
    for batch_directory in sorted(
        replication_subdirectories, key=lambda subdir: int(subdir.split("_")[-1])
    ):
        full_batch_directory = os.path.join(replication_directory, batch_directory)
        for iteration in os.listdir(full_batch_directory):
            try:
                # Visualize codex_query_results.json for each iteration of this batch size.
                codex_results_file = os.path.join(
                    full_batch_directory,
                    iteration,
                    CodexSampleGenerator.query_results_file,
                )
                print(codex_results_file)
                assert os.path.exists(codex_results_file)
                with open(codex_results_file) as f:
                    codex_results = json.load(f)
                # Montage the prompted programs.
                prompted_programs = get_prompted_programs(codex_results)
                if len(prompted_programs) > 0:
                    prompted_progams_montage = drawings_primitives.display_programs_as_grid(
                        prompted_programs,
                        color=(0, 0, 0),
                        suptitle=f"Codex prompt program examples: {batch_directory}, stitch_iteration={iteration}",
                        transparent_background=False,
                        ncols=8,
                    )
                    codex_results_figure.append(prompted_progams_montage)

                # Montage the sampled programs.
                sampled_programs = get_sampled_programs(codex_results)
                if len(sampled_programs) > 0:
                    sampled_programs_montage = drawings_primitives.display_programs_as_grid(
                        sampled_programs,
                        color=(0, 0, 255),
                        suptitle=f"Codex program samples: {batch_directory}, stitch_iteration={iteration}",
                        transparent_background=False,
                        ncols=8,
                    )

                    codex_results_figure.append(sampled_programs_montage)
                # Add a line spacer.

                codex_results_figure.append(get_blank_spacer(codex_results_figure))

            except Exception as e:
                print(e)
                continue

    # Concatenate all the images and save it.
    codex_visualization_output = os.path.join(
        replication_directory, CODEX_SAMPLE_VISUALIZATION
    )
    print(f"Writing out visualization to: {codex_visualization_output}")
    codex_results_image = stack_images(codex_results_figure)
    codex_results_image.save(codex_visualization_output)


def main(args):
    replication_directories = get_replication_directories(args)
    for (replication_directory, replication_subdirectories) in replication_directories:
        print(f"Now generating visualizations for replication: {replication_directory}")
        if args.visualize_codex_results:
            visualize_codex_results(
                args, replication_directory, replication_subdirectories
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)