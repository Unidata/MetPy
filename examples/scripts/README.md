This directory contains scripts generated from the example notebooks. The
scripts can be regenerated automatically by running:

`ipython nbconvert`

This will pick up the configuration from the `ipython_nbconvert_config.py`
file in this directory. The output is controlled by the `jinja` template
in this directory (`python-scripts.tpl`). #Note:# This file is very sensitive
to newlines you add. Don't add any standalone scripts
here, but rather add a notebook and use `nbconvert` to make scripts.
