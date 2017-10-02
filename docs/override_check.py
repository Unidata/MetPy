#!/usr/lib/env python
#
# Module Documentation Override Check
#
# Verifies that any modules present in the _templates/overrides directory have all of their
# their exported functions included in the doc file.

from __future__ import print_function

import glob
import importlib
import os
import sys


failed = False
for full_path in glob.glob('_templates/overrides/metpy.*.rst'):

    filename = os.path.basename(full_path)
    module = filename.split('.rst')[0]

    # Get all functions in the module
    i = importlib.import_module(module)
    functions = set(i.__all__)

    # Get all lines in the file
    with open(full_path) as f:
        lines = f.read().splitlines()
    lines = {line.strip() for line in lines}

    # Check for any missing functions
    missing_functions = functions - lines

    if missing_functions:
        failed = True
        print('ERROR - The following functions are missing from the override file ' +
              filename + ': ' + ', '.join(missing_functions), file=sys.stderr)

# Report status
if failed:
    sys.exit(1)
else:
    print('Override check successful.')
