# Copyright (c) 2021 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
Custom flake8 plugin to catch MetPy-specific bad style/practice.

Currently this only looks for multiplication or division by units, since that can break
masked arrays and is slower than calling ``Quantity()``.
"""

import ast
from collections import namedtuple

Error = namedtuple('Error', 'lineno col code')


class MetPyVisitor(ast.NodeVisitor):
    """Visit nodes of the AST looking for violations."""

    def __init__(self):
        """Initialize the visitor."""
        self.errors = []

    @staticmethod
    def _is_unit(node):
        """Check whether a node should be considered to represent "units"."""
        # Looking for a .units attribute, a units.myunit, or a call to units()
        is_units_attr = isinstance(node, ast.Attribute) and node.attr == 'units'
        is_reg_attr = (isinstance(node, ast.Attribute)
                       and isinstance(node.value, ast.Name) and node.value.id == 'units')
        is_reg_call = (isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == 'units')
        is_unit_alias = isinstance(node, ast.Name) and 'unit' in node.id

        return is_units_attr or is_reg_attr or is_reg_call or is_unit_alias

    def visit_BinOp(self, node):  # noqa: N802
        """Visit binary operations."""
        # Check whether this is multiplying or dividing by units
        if (isinstance(node.op, (ast.Mult, ast.Div))
                and (self._is_unit(node.right) or self._is_unit(node.left))):
            self.error(node.lineno, node.col_offset, 1)

        super().generic_visit(node)

    def error(self, lineno, col, code):
        """Add an error to our output list."""
        self.errors.append(Error(lineno, col, code))


class MetPyChecker:
    """Flake8 plugin class to check MetPy style/best practice."""

    name = __name__
    version = '1.0'

    def __init__(self, tree):
        """Initialize the plugin."""
        self.tree = tree

    def run(self):
        """Run the plugin and yield errors."""
        visitor = MetPyVisitor()
        visitor.visit(self.tree)
        for err in visitor.errors:
            yield self.error(err)

    def error(self, err):
        """Format errors into Flake8's required format."""
        return (err.lineno, err.col,
                f'MPY{err.code:03d}: Multiplying/dividing by units--use units.Quantity()',
                type(self))
