# setup_osx.sh: set up script | Author: Catherine Wong.
# General setup script to install dependencies for this repository.
# Download LAPS-DreamCoder submodule dependency, which needs to be unpacked into laps/dreamcoder
git submodule update --init --recursive

# Install python requirements. N.B. this is tested using Python 3.7.7
pip3 install -r laps_requirements.txt

# Install the NLTK word tokenize package.
python -m nltk.downloader 'punkt'

# Install the spacy word2vec model.
 python -m spacy download en_core_web_lg

# TODO: install the transformers.