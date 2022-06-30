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


def get_added():
    """Get all lines added in the most recent merge."""
    revs = subprocess.check_output(['git', 'rev-list', '--parents', '-n', '1', 'HEAD'])
    merge_commit, target, _ = revs.decode('utf-8').split()
    diff = subprocess.check_output(['git', 'diff', '{}...{}'.format(target, merge_commit)])
    return '\n'.join(line for line in diff.decode('utf-8').split('\n')
                     if line.startswith('+') and not line.startswith('+++'))


if __name__ == '__main__':
    # If the second argument is true, then we only want links in the most recent merge,
    # otherwise we print all failing links.
    if sys.argv[2] in ('true', 'True'):
        print('Checking only links in the diff')
        added = get_added()
        check_link = lambda l: l['uri'] in added
    else:
        print('Checking all links')
        check_link = lambda l: True

    ret = 0
    for link in get_failing_links(sys.argv[1]):
        if check_link(link):
            ret = 1
            print(f'{link["filename"]}:{link["lineno"]}: {link["uri"]} -> '
                  f'{link["status"]} {link["info"]}')

    sys.exit(ret)
