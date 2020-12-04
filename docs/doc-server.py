#!/usr/bin/env python
# Copyright (c) 2009,2017,2018,2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Start a server for hosting the built HTML documentation."""

from functools import partial
import http.server
import pathlib
import posixpath
import socketserver
import sys

PORT = 8000
TEST_FILES_DIR = pathlib.Path('test-server')


class Server(http.server.SimpleHTTPRequestHandler):
    """Server handles serving docs by dynamically remapping to the build directory."""

    def translate_path(self, path):
        """Translate a request path to the proper path into the built docs."""
        if path == '/MetPy/banner.html':
            return str(TEST_FILES_DIR / 'banner.html')
        elif path == '/MetPy/versions.json':
            return str(TEST_FILES_DIR / 'versions.json')
        elif path.startswith('/MetPy/'):
            path = posixpath.join('/', *path.split('/')[3:])
        return super().translate_path(path)


build_server = partial(Server, directory='build/html')

with socketserver.TCPServer(('', PORT), build_server) as httpd:
    try:
        print(f'Serving docs at: http://localhost:{PORT}/MetPy/v1.0')
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
