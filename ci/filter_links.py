#!/usr/bin/env python
# Copyright (c) 2021 MetPy Developers.
"""Filter links from Sphinx's linkcheck."""
import json
import subprocess
import sys


def get_failing_links(fname):
    """Yield links with problematic statuses."""
    with open(fname, 'rt') as linkfile:
        links = json.loads('[' + ','.join(linkfile) + ']')
        for link in links:
            if link['status'] not in {'working', 'ignored', 'unchecked'}:
                yield link


def get_added(merge_commit, target):
    """Get all lines added between start and end git ids."""
    diff = subprocess.check_output(['git', 'diff', '{}...{}'.format(target, merge_commit)])
    return '\n'.join(line for line in diff.decode('utf-8').split('\n')
                     if line.startswith('+') and not line.startswith('+++'))


if __name__ == '__main__':
    # If we have the args to get a diff, then we only want links from that diff, otherwise
    # we print all failing links.
    if len(sys.argv) >= 4:
        added = get_added(sys.argv[2], sys.argv[3])
        check_link = lambda l: l['uri'] in added
    else:
        check_link = lambda l: True

    ret = 0
    for link in get_failing_links(sys.argv[1]):
        if check_link(link):
            ret = 1
            print(f'{link["filename"]}:{link["lineno"]}: {link["uri"]} -> '
                  f'{link["status"]} {link["code"]}')

    sys.exit(ret)
