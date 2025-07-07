#!/bin/bash
git config --global --add safe.directory /container-benchmarks
git config --global --add safe.directory /container-benchmarks/.git
cd /container-benchmarks/benchmarks
./asv_run_script.sh
