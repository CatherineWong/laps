# setup_sketch3.sh: set up script | Author: Catherine Wong.
# Setup script for sketch3 CSAIL machine. Should be run upon cloning the directory. 

# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Create a new Conda environment called `laps` with Python 3.7.7
conda env create -f environment.yml
conda activate laps

# Install the NLTK word tokenize package.
python -m nltk.downloader 'punkt'

# Install Rust for stitch.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Use the linux-specific OCaml binaries.
rm ocaml/bin/*
cp ocaml/linux_bin/* ocaml/bin/*