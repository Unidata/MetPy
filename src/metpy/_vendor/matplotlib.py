# Copyright (c) 2019 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""Vendor core functionality used from matplotlib.

This code has been reproduced from matplotlib 3.3.4 in accord with its license agreement
(reproduced below).

    1. This LICENSE AGREEMENT is between the Matplotlib Development Team ("MDT"), and the
    Individual or Organization ("Licensee") accessing and otherwise using matplotlib software
    in source or binary form and its associated documentation.

    2. Subject to the terms and conditions of this License Agreement, MDT hereby grants
    Licensee a nonexclusive, royalty-free, world-wide license to reproduce, analyze, test,
    perform and/or display publicly, prepare derivative works, distribute, and otherwise use
    matplotlib 3.3.4 alone or in any derivative version, provided, however, that MDT's License
    Agreement and MDT's notice of copyright, i.e., "Copyright (c) 2012-2013 Matplotlib
    Development Team; All Rights Reserved" are retained in matplotlib 3.3.4 alone or in any
    derivative version prepared by Licensee.

    3. In the event Licensee prepares a derivative work that is based on or incorporates
    matplotlib 3.3.4 or any part thereof, and wants to make the derivative work available to
    others as provided herein, then Licensee hereby agrees to include in any such work a brief
    summary of the changes made to matplotlib 3.3.4.

    4. MDT is making matplotlib 3.3.4 available to Licensee on an "AS IS" basis. MDT MAKES NO
    REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF EXAMPLE, BUT NOT LIMITATION,
    MDT MAKES NO AND DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
    FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF MATPLOTLIB 3.3.4 WILL NOT INFRINGE ANY THIRD
    PARTY RIGHTS.

    5. MDT SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB 3.3.4 FOR ANY
    INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF MODIFYING,
    DISTRIBUTING, OR OTHERWISE USING MATPLOTLIB 3.3.4, OR ANY DERIVATIVE THEREOF, EVEN IF
    ADVISED OF THE POSSIBILITY THEREOF.

    6. This License Agreement will automatically terminate upon a material breach of its terms
    and conditions.

    7. Nothing in this License Agreement shall be deemed to create any relationship of agency,
    partnership, or joint venture between MDT and Licensee. This License Agreement does not
    grant permission to use MDT trademarks or trade name in a trademark sense to endorse or
    promote products or services of Licensee, or any third party.

    8. By copying, installing or otherwise using matplotlib 3.3.4, Licensee agrees to be bound
    by the terms and conditions of this License Agreement.
"""
import matplotlib.transforms as mtransforms


class _TransformedBoundsLocator:
    """
    Copyright (c) 2012-2013 Matplotlib Development Team; All Rights Reserved

    This class is reproduced exactly from matplotlib/axes/_base.py, excluding the
    modifications made to this comment.

    Axes locator for `.Axes.inset_axes` and similarly positioned axes.
    The locator is a callable object used in `.Axes.set_aspect` to compute the
    axes location depending on the renderer.
    """

    def __init__(self, bounds, transform):
        """
        *bounds* (a ``[l, b, w, h]`` rectangle) and *transform* together
        specify the position of the inset axes.
        """
        self._bounds = bounds
        self._transform = transform

    def __call__(self, ax, renderer):
        # Subtracting transSubfigure will typically rely on inverted(),
        # freezing the transform; thus, this needs to be delayed until draw
        # time as transSubfigure may otherwise change after this is evaluated.
        return mtransforms.TransformedBbox(
            mtransforms.Bbox.from_bounds(*self._bounds),
            self._transform - ax.figure.transSubfigure)
