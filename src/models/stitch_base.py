"""
stitch_base.py | Author: Gabe Grand.
Base class containing utilities for working with the Stitch library.

https://github.com/mlb2251/stitch
"""

import json
import os
import subprocess


class StitchBase(object):
    def run_binary(
        self,
        bin: str = "compress",
        stitch_args: list = [],
        stitch_kwargs: dict = {},
    ):
        """Calls `cargo run` to invoke Stitch via subprocess call.

        params:
            bin: Stitch binary.
            stitch_args: Positional arguments to Stitch CLI.
            stitch_kwargs: Keyword arguments to pass to Stitch CLI.

        """
        assert stitch_args or stitch_kwargs
        stitch_base_command = (
            f"cd stitch; cargo run --bin={bin} --release -- {' '.join(stitch_args)} "
        )
        stitch_command = stitch_base_command + " ".join(
            [f"--{k}={v}" for k, v in stitch_kwargs.items()]
        )
        print("Running Stitch with the following command:")
        print(stitch_command)

        subprocess.run(stitch_command, capture_output=True, check=True, shell=True)

    def write_frontiers_to_file(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        frontiers_filename: str = "stitch_frontiers.json",
    ):
        """Dumps programs from frontiers to a file that can be passed to Stitch.

        returns:
            Path to JSON file containing a list of programs.
        """
        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=False,
        )
        programs = []
        for split in frontiers:
            for frontier in frontiers[split]:
                frontier_programs = [
                    str(entry.program).replace("lambda", "lam") for entry in frontier
                ]
                programs += frontier_programs
        # Write out the programs.
        frontiers_filepath = os.path.join(
            experiment_state.get_checkpoint_directory(),
            frontiers_filename,
        )
        with open(frontiers_filepath, "w") as f:
            json.dump(programs, f)
        return frontiers_filepath

    def get_inventions_from_file(self, stitch_output_file: str):
        with open(stitch_output_file, "r") as f:
            stitch_results = json.load(f)

        inv_name_to_dc_fmt = {
            inv["name"]: inv["dreamcoder"] for inv in stitch_results["invs"]
        }

        # Replace `inv0` with inlined definitions in dreamcoder format
        for inv_name, inv_dc_fmt in inv_name_to_dc_fmt.items():
            for prior_inv_name, prior_inv_dc_fmt in inv_name_to_dc_fmt.items():
                # Assume ordered dict with inventions inv0, inv1, ...
                # inv_i only includes prior inventions inv0, ..., inv_i-1
                if prior_inv_name == inv_name:
                    break
                inv_dc_fmt.replace(prior_inv_name, prior_inv_dc_fmt)
            inv_name_to_dc_fmt[inv_name] = inv_dc_fmt

        return inv_name_to_dc_fmt