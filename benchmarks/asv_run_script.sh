#!/bin/bash
#Run asv

#Set up asv machine
asv machine --yes

# Runs asv on the commits in the hash file but skips ones that already have results
asv run --skip-existing-successful HASHFILE:no_bot_merge_commits.txt
