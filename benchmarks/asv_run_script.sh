#!/bin/bash
#Run asv

# Generate artificial data file for benchmarks
python3 data_array_generate.py

#Set up asv machine
asv machine --yes

# Runs asv on the commits in the hash file but skips ones that already have results
asv run --skip-existing-successful HASHFILE:no_bot_merge_commits.txt
