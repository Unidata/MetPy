=======================
GEMPAK Conversion Guide
=======================

The popular meteorological plotting and analysis tool GEMPAK is no longer formally supported
and Unidata is only producing occasional maintenance releases. To help GEMPAK users convert to
the Python ecosystem we have compiled this list of GEMPAK programs, commands, parameters, etc.
and their Python equivalents. These tables are continually updated as we add more functionality
to MetPy as part of an National Science Foundation grant. Where issues have been filed, we have
linked to them. Please feel free to tackle of these, comment on their importance in the issue
tracker, or file a new issue.

Green rows represent implemented functionality, red is not implemented (with issue listed),
blue is uncertain of parity, and white is unevaluated.

.. raw:: html

   <style type="text/css">
    .wy-table-responsive {border-style:solid; border-width:1px;}
    .wy-table-responsive td, th{border-style:solid;border-width:1px;}
    .wy-table-responsive .tg-implemented{background-color: #D9EAD3}
    .wy-table-responsive .tg-notimplemented{background-color: #F4CDCD}
    .wy-table-responsive .tg-yes{background-color: #93C47D}
    .wy-table-responsive .tg-no{background-color: #E06666}
    .wy-table-responsive .tg-warning{background-color: #FFD966}
    .wy-table-responsive .tg-info{background-color: #D0E3F3}
    </style>

    <h2>GEMPAK Scalar Output/Scalar Grid</h2>
    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equivalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td class="tg-implemented">ABS(S)</td>
        <td class="tg-implemented">Absolute value</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.absolute.html#numpy.absolute">numpy.absolute</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">ACOS(S)</td>
        <td class="tg-implemented">Arc cosine</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.arccos.html#numpy.arccos">numpy.arccos</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">ASIN(S)</td>
        <td class="tg-implemented">Arc sine</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.arcsin.html#numpy.arcsin">numpy.arcsin</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">ATAN(S)</td>
        <td class="tg-implemented">Arc tangent</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan.html#numpy.arctan">numpy.arctan</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">ATN2(S1, S2)</td>
        <td class="tg-implemented">Arc tangent</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html#numpy.arctan2">numpy.arctan2</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">COS(S)</td>
        <td class="tg-implemented">Cosine</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.cos.html#numpy.cos">numpy.cos</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">EXP(S1, S2)</td>
        <td class="tg-implemented">Exponential to real</td>
        <td class="tg-implemented">S1**S2</td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">EXPI(S1, S2)</td>
        <td class="tg-implemented">Exponential to integer (uses NINT)</td>
        <td class="tg-implemented">S1 ** S2.<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.chararray.astype.html#numpy.chararray.astype">astype(int)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">INT(S)</td>
        <td class="tg-implemented">Convert to integer</td>
        <td class="tg-implemented">S.<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.chararray.astype.html#numpy.chararray.astype">astype(int)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-no">No</td>
      </tr>
      <tr>
        <td class="tg-implemented">LN(S)</td>
        <td class="tg-implemented">Natural logarithm</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html#numpy.log">numpy.log</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">LOG(S)</td>
        <td class="tg-implemented">Base 10 logarithm</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html#numpy.log10">numpy.log10</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">NINT(S)</td>
        <td class="tg-implemented">Round to nearest integer</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.round_.html#numpy.round_">numpy.round</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">SIN(S)</td>
        <td class="tg-implemented">Sine</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sin.html#numpy.sin">numpy.sin</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SQRT(S)</td>
        <td class="tg-implemented">Square root</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sqrt.html#numpy.sqrt">numpy.sqrt</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">TAN(S)</td>
        <td class="tg-implemented">Tangent</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.tan.html#numpy.tan">numpy.tan</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">NCDF(S1, S2, S3)</td>
        <td class="tg-info">Cumulative normal distribution for value, mean, std dev</td>
        <td class="tg-info"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html">scipy.stats.norm.cdf</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">INCD(S1, S2, S3)</td>
        <td class="tg-info">Inverse cumulative normal distribution value given cumulative probability, mean, std dev</td>
        <td class="tg-info"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html">scipy.stats.norm.ppf(scipy.stats.norm.cdf</a>)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">BNCDF(S1, S2, S3, S4)</td>
        <td class="tg-info">Cumulative binormal mixture distribution given value, mode, left sigma, and right sigma</td>
        <td class="tg-info"><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html">scipy.stats.multivariate_normal</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IBNCDF(S1, S2, S3, S4)</td>
        <td>Inverse cumulative binormal mixture distribution value given cumulative probability, mode, left sigma, and right sigma</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PAUB(S1, S2, S3)</td>
        <td>Combine probabilities (S1 or S2) with dependency parameter S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">ADD(S1, S2)</td>
        <td class="tg-implemented">Addition</td>
        <td class="tg-implemented">S1 + S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MUL(S1, S2)</td>
        <td class="tg-implemented">Multiplication</td>
        <td class="tg-implemented">S1 * S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">QUO(S1, S2)</td>
        <td class="tg-implemented">Division</td>
        <td class="tg-implemented">S1 / S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SUB(S1, S2)</td>
        <td class="tg-implemented">Subtraction</td>
        <td class="tg-implemented">S1 - S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>BOOL(S)</td>
        <td>Boolean function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SLT(S1, S2)</td>
        <td class="tg-implemented">Less than function</td>
        <td class="tg-implemented">S1 &lt; S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SLE(S1, S2)</td>
        <td class="tg-implemented">Less than or equal to function</td>
        <td class="tg-implemented">S1 &lt;= S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SGT(S1, S2)</td>
        <td class="tg-implemented">Greater than function</td>
        <td class="tg-implemented">S1 &gt; S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SGE(S1, S2)</td>
        <td class="tg-implemented">Greater than or equal to function</td>
        <td class="tg-implemented">S1 &gt;= S2</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SBTW(S1, S2, S3)</td>
        <td class="tg-implemented">Between two values function</td>
        <td class="tg-implemented">S1 &gt; S2 and S1 &lt; S3</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>SMAX(S1, S2)</td>
        <td>Maximum of S1 and S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMIN(S1, S2)</td>
        <td>Minimum of S1 and S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MASK(S1, S2)</td>
        <td>Mask S1 based on S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MISS(S1, S2)</td>
        <td>Set missing values in S1 to S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">ADV(S, V)</td>
        <td class="tg-implemented">Advection</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.advection.html#metpy.calc.advection">metpy.calc.advection</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">AVG(S1, S2)</td>
        <td class="tg-implemented">Average</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean">numpy.mean</a>(<a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html">np.array</a>([S1, S2]), axis=0)</td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-no">No</td>
      </tr>
      <tr>
        <td class="tg-implemented">AVOR(V)</td>
        <td class="tg-implemented">Absolute vorticity</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.absolute_vorticity.html">metpy.calc.absolute_vorticity</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">BVSQ(THTA)</td>
        <td class="tg-implemented">Brunt-Vaisala frequency squared in a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.brunt_vaisala_frequency_squared.html#metpy.calc.brunt_vaisala_frequency_squared">metpy.calc.brunt_vaisala_frequency_squared</a> (use with <a href="api/generated/metpy.calc.mixed_layer.html#metpy.calc.mixed_layer">metpy.calc.mixed_layer</a> to obtain layer average)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">CROS(V1, V2)</td>
        <td class="tg-implemented">Vector cross product magnitude</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.cross.html#numpy.cross">numpy.cross</a>(V1, V2)</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDEN(PRES, TMPC)</td>
        <td class="tg-implemented">Density of dry air</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.density.html#metpy.calc.density">metpy.calc.density(mixing=0)</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDR(S)</td>
        <td class="tg-implemented">Partial derivative with respect to R</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDT(S)</td>
        <td class="tg-implemented">Time derivative</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDX(S)</td>
        <td class="tg-implemented">Partial derivative with respect to X</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDY(S)</td>
        <td class="tg-implemented">Partial derivative with respect to Y</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DEF(V)</td>
        <td class="tg-implemented">Total deformation</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.total_deformation.html#metpy.calc.total_deformation">metpy.calc.total_deformation</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>DIRN(V)</td>
        <td>Direction relative to north</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DIRR(V)</td>
        <td>Direction relative to grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DIV(V)</td>
        <td class="tg-implemented">Divergence</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.h_divergence.html#metpy.calc.h_divergence">metpy.calc.h_divergence</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>DIVT(S, V)</td>
        <td>Divergence tendency (only for cylindrical grids)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DOT(V1, V2)</td>
        <td class="tg-implemented">Vector dot product</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html#numpy.dot">numpy.dot</a>(V1, V2)</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DTH(S)</td>
        <td class="tg-implemented">Partial derivative with respect to theta</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">FCNT(S)</td>
        <td class="tg-info">Coriolis force at grid center (polar)</td>
        <td class="tg-info"><a href="api/generated/metpy.calc.coriolis_parameter.html#metpy.calc.coriolis_parameter">metpy.calc.coriolis_parameter</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">FOSB(TMPC, RELH, SPED)</td>
        <td class="tg-notimplemented">Fosberg index (Fire weather index frontogenesis)</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/636">Issue #636</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">FRNT(THETA, V)</td>
        <td class="tg-implemented">Frontogenesis</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.frontogenesis.html#metpy.calc.frontogenesis">metpy.calc.frontogenesis</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">GWFS(S, N)</td>
        <td class="tg-implemented">Filter with normal distribution of weights</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.smooth_gaussian.html#metpy.calc.smooth_gaussian">metpy.calc.smooth_gaussian</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-yes">Yes</td>
        <td></td>
      </tr>
      <tr>
        <td>RDFS(S, DX)</td>
        <td>GWFS applied for given output resolution</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HIGH(S, RADIUS)</td>
        <td>Relative maxima over a grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>JCBN(S1, S2)</td>
        <td>Jacobian determinant</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">KNTS(S)</td>
        <td class="tg-implemented">Convert m/s to knots</td>
        <td class="tg-implemented">.to('knots')</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">LAP(S)</td>
        <td class="tg-implemented">Laplacian operator</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.laplacian.html#metpy.calc.laplacian">metpy.calc.laplacian</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>LAV(S)</td>
        <td>Layer average (2-levels)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LDF(S)</td>
        <td>Layer difference (2-levels)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LOWS(S, RADIUS)</td>
        <td>Relative minima over a grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MAG(V)</td>
        <td class="tg-implemented">Magnitude of a vector</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm">numpy.linalg.norm</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>MASS</td>
        <td>Mass per unit volume in a layer from PRES</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MDIV(V)</td>
        <td>Layer-avg. mass divergence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MIXR(DWPC, PRES)</td>
        <td class="tg-implemented">Mixing ratio</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.mixing_ratio.html#metpy.calc.mixing_ratio">metpy.calc.mixing_ratio</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">MRAD(V, LAT, LON, DIR, SPD)</td>
        <td class="tg-notimplemented">Magnitude of radial wind</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/653">Issue #653</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">MSDV(S, V)</td>
        <td class="tg-notimplemented">Layer-avg. mass-scalar flux divergence</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/655">Issue #655</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MSFC(V)</td>
        <td class="tg-implemented">Pseudo angular momentum (cross-sections)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.absolute_momentum.html#metpy.calc.absolute_momentum">metpy.calc.absolute_momentum</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">MTNG(V, LAT, LON, DIR, SPD)</td>
        <td class="tg-notimplemented">Magnitude of tangential wind</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/653">Issue #653</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">NORM(V)</td>
        <td class="tg-implemented">Normal component (cross-sections)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.normal_component.html#metpy.calc.normal_component">metpy.calc.normal_component</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>NMAX(S, ROI)</td>
        <td>Neighborhood maximum value for a radius of influence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NMIN(S, ROI)</td>
        <td>Neighborhood minimum value for a radius of influence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PLAT(S)</td>
        <td>Latitude at each point (polar)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PLON(S)</td>
        <td>Longitude at each point (polar)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">PLCL(PRES, TMPC, DWPC)</td>
        <td class="tg-implemented">Pressure of the lifting condensation level</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html#metpy.calc.lcl">metpy.calc.lcl</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">POIS(S1, S2)</td>
        <td class="tg-notimplemented">Solve Poisson equation of forcing function with boundary conditions</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/651">Issue #651</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">POLF(S)</td>
        <td class="tg-info">Coriolis force at each point (polar)</td>
        <td class="tg-info"><a href="api/generated/metpy.calc.coriolis_parameter.html#metpy.calc.coriolis_parameter">metpy.calc.coriolis_parameter</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">PVOR(S, V)</td>
        <td class="tg-implemented">Potential vorticity in a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.potential_vorticity_baroclinic.html#metpy.calc.potential_vorticity_baroclinic">metpy.calc.potential_vorticity_baroclinic</a>
                                   <br><a href="api/generated/metpy.calc.potential_vorticity_baroclinic.html#metpy.calc.potential_vorticity_barotropic">metpy.calc.potential_vorticity_barotropic</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">PVR3(S, V)</td>
        <td class="tg-notimplemented">3-D potential vorticity for a level</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/649">Issue #649</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">RELH(TMPC, DWPT)</td>
        <td class="tg-implemented">Relative humidity</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.relative_humidity_from_dewpoint.html#metpy.calc.relative_humidity_from_dewpoint">metpy.calc.relative_humidity_from_dewpoint</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">RICH(V)</td>
        <td class="tg-notimplemented">Richardson stability number in a layer</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/647">Issue #647</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">ROSS(V1, V2)</td>
        <td class="tg-notimplemented">Rossby number</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/637">Issue #637</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">SAVG(S)</td>
        <td class="tg-info">Average over whole grid</td>
        <td class="tg-info"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean">numpy.mean</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">SAVS(S)</td>
        <td class="tg-info">Average over display subset area of grid</td>
        <td class="tg-info"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean">numpy.mean</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">SDIV(S, V)</td>
        <td class="tg-notimplemented">Flux divergence of a scalar</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/646">Issue #646</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SGMX(S)</td>
        <td class="tg-implemented">Maximum of S over GAREA</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmax.html#numpy.nanmax">numpy.nanmax</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SGMN(S)</td>
        <td class="tg-implemented">Minimum of S over GAREA</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.nanmin.html#numpy.nanmin">numpy.nanmin</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SHR(V)</td>
        <td class="tg-implemented">Shearing deformation</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.shearing_deformation.html#metpy.calc.shearing_deformation">metpy.calc.shearing_deformation</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SM5S(S)</td>
        <td class="tg-implemented">5-point smoother</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.smooth_n_point.html#metpy.calc.smooth_n_point">metpy.calc.smooth_n_point</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SM9S(S)</td>
        <td class="tg-implemented">9-point smoother</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.smooth_n_point.html#metpy.calc.smooth_n_point">metpy.calc.smooth_n_point</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>STAB(TMPC)</td>
        <td>Lapse rate over a layer in K/km</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">STR(V)</td>
        <td class="tg-implemented">Stretching deformation</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.stretching_deformation.html#metpy.calc.stretching_deformation">metpy.calc.stretching_deformation</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TANG(V)</td>
        <td class="tg-implemented">Tangential component (cross-sections)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.tangential_component.html#metpy.calc.tangential_component">metpy.calc.tangential_component</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>TAV(S)</td>
        <td>Time average</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDF(S)</td>
        <td>Time difference</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">THES(PRES, TMPC)</td>
        <td class="tg-implemented">Saturated equivalent potential temperature</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.saturation_equivalent_potential_temperature.html#metpy.calc.saturation_equivalent_potential_temperature">metpy.calc.saturation_equivalent_potential_temperature</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">THTA(TMPC, PRES)</td>
        <td class="tg-implemented">Potential temperature</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.potential_temperature.html#metpy.calc.potential_temperature">metpy.calc.potential_temperature</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">THTE(PRES, TMPC, DWPC)</td>
        <td class="tg-implemented">Equivalent potential temperature</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.equivalent_potential_temperature.html">metpy.calc.equivalent_potential_temperature</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">THWC(PRES, TMPC, DWPC)</td>
        <td class="tg-implemented">Wet bulb temperature</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.wet_bulb_temperature.html">metpy.calc.wet_bulb_temperature</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TLCL(TMPC, DWPC)</td>
        <td class="tg-implemented">Temperature of the lifting condensation level</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html">metpy.calc.lcl</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TMST(THTE, PRES)</td>
        <td class="tg-implemented">Parcel temperature along a moist adiabat</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.moist_lapse.html#metpy.calc.moist_lapse">metpy.calc.moist_lapse</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TMWK(PRES, TMPK, RMIX)</td>
        <td class="tg-implemented">Web bulb temperature in Kelvin</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.wet_bulb_temperature.html">metpy.calc.wet_bulb_temperature</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>DSUB(V1, V2)</td>
        <td>DIRN(V1) - DIRN(V2)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>UN(V)</td>
        <td>North relative u component</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>UR(V)</td>
        <td>Grid relative u component</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VN(V)</td>
        <td>North relative v component</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">VOR(V)</td>
        <td class="tg-implemented">Vorticity</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.v_vorticity.html#metpy.calc.v_vorticity">metpy.calc.v_vorticity</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>VR(V)</td>
        <td>Grid relative v component</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">WNDX(S1, S2, S3, S4)</td>
        <td class="tg-notimplemented">WINDEX (index for microburst potential)</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/639">Issue #639</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSHR(V)</td>
        <td>Magnitude of the vertical wind shear in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">XAV(S)</td>
        <td class="tg-implemented">Average along a display subset grid row</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean">numpy.mean(a, axis=1)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">XSUM(S)</td>
        <td class="tg-implemented">Sum along a display subset grid row</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html#numpy.sum">numpy.sum(a, axis=1)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">YAV(S)</td>
        <td class="tg-implemented">Average along a display subset grid column</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html#numpy.mean">numpy.mean(a, axis=0)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
      <tr>
        <td class="tg-implemented">YSUM(S)</td>
        <td class="tg-implemented">Sum along a display subset grid column</td>
        <td class="tg-implemented"><a href="https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html#numpy.sum">numpy.sum(a, axis=0)</a></td>
        <td class="tg-yes">Yes</td>
        <td class="tg-no">No</td>
        <td class="tg-yes">Yes</td>
      </tr>
    </table>


.. raw:: html

    <h2>Vector Output Grid</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th class="tg=yw4l">Units?</th>
      </tr>
      <tr>
        <td class="tg-implemented">AGE(S)</td>
        <td class="tg-implemented">Ageostrophic wind</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.ageostrophic_wind.html#metpy.calc.ageostrophic_wind">metpy.calc.ageostrophic_wind</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">CIRC(V, S)</td>
        <td class="tg-notimplemented">Circulation (cross-section)</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/662">Issue #662</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DVDX(V)</td>
        <td class="tg-implemented">Partial x derivative of V</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DVDY(V)</td>
        <td class="tg-implemented">Partial y derivative of V</td>
        <td class="tg-implemented"><a href="api/generated/generated/metpy.calc.first_derivative.html">metpy.calc.first_derivative</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">GCIR(LAT, LON)</td>
        <td class="tg-notimplemented">Great circle from point to antipodal point</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">GCWV(LAT, LON, MASK)</td>
        <td class="tg-notimplemented">Great circle from point to antipodal point with land blocking algorithm MASK will usually be SEA</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">GEO(S)</td>
        <td class="tg-implemented">Geostrophic wind</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.geostrophic_wind.html#metpy.calc.geostrophic_wind">metpy.calc.geostrophic_wind</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">GRAD(S)</td>
        <td class="tg-implemented">Gradient of a scalar</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.gradient.html#metpy.calc.gradient">metpy.calc.gradient</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">GWFV(V, N)</td>
        <td class="tg-notimplemented">Filter with normal distribution of weights</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/664">Issue #664</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RDFV(V, DX)</td>
        <td>GWFV applied for given output resolution</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">INAD(V1, V2)</td>
        <td class="tg-implemented">Inertial advective wind</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.inertial_advective_wind.html">metpy.calc.inertial_advective_wind</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">ISAL(S)</td>
        <td class="tg-notimplemented">Isallobaric wind</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/641">Issue #641</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>KCRS(V)</td>
        <td>Curl of V</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">KNTV(V)</td>
        <td class="tg-implemented">Convert m/s to knots</td>
        <td class="tg-implemented">.to('knots')</td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">LTRN(S, V)</td>
        <td class="tg-notimplemented">Layer-averaged transport of a scalar</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">NORMV(V)</td>
        <td class="tg-implemented">Vector normal wind (cross-section)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.normal_component.html#metpy.calc.normal_component">metpy.calc.normal_component</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">QVEC(S, V)</td>
        <td class="tg-implemented">Q-vector at a level</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.q_vector.html">metpy.calc.q_vector</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">QVCL(THTA, V)</td>
        <td class="tg-implemented">Q-vector of a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.q_vector.html#metpy.calc.q_vector">metpy.calc.q_vector</a> (use with <a href="api/generated/metpy.calc.mixed_layer.html#metpy.calc.mixed_layer">metpy.calc.mixed_layer</a> to obtain layer average of each component)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">RAD(V, LAT, LON, DIR, SPD)</td>
        <td class="tg-notimplemented">Storm relative radial wind</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ROT(angle, V)</td>
        <td>Coordinate rotation</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMUL(S, V)</td>
        <td>Multiply a vector's components by a scalar</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">SM5V(V)</td>
        <td class="tg-implemented">5-point smoother</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.smooth_n_point.html#metpy.calc.smooth_n_point">metpy.calc.smooth_n_point</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>SQUO(S, V)</td>
        <td>Vector division by a scalar</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TANGV(V)</td>
        <td class="tg-implemented">Vector tangential wind (cross-section)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.tangential_component.html#metpy.calc.tangential_component">metpy.calc.tangential_component</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">THRM(S)</td>
        <td class="tg-notimplemented">Thermal wind over a layer</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">TNG(V, LAT, LON, DIR,SPD)</td>
        <td class="tg-notimplemented">Storm relative tangential wind</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/658">Issue #658</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VADD(V1, V2)</td>
        <td>Add the components of two vectors</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VASV(V1, V2)</td>
        <td>Vector component of V1 along V2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VAVG(V)</td>
        <td>Average over whole grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VAVS(V)</td>
        <td>Average vector over display subset grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VECN(S1, S2)</td>
        <td>Create vector from north relative components</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VECR(S1, S2)</td>
        <td>Create vector from grid relative components</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VESD(SPD, DIR)</td>
        <td>Create vector from speed and north-rel direction</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VLAV(V)</td>
        <td>Layer average for a vector</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VLDF(V)</td>
        <td>Layer difference for a vector</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VMUL(V1, V2)</td>
        <td>Multiple the components of two vectors</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VQUO(V1, V2)</td>
        <td>Divide the components of two vectors</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSUB(V1, V2)</td>
        <td>Subtract the components of two vectors</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VLT(V, S)</td>
        <td>Less than function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VLE(V, S)</td>
        <td>Less than or equal to function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VGT(V, S)</td>
        <td>Greater than function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VGE(V, S)</td>
        <td>Greater than or equal to function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VBTW(V, S1, S2)</td>
        <td>Between two values function</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VMSK(V, S)</td>
        <td>Mask V based on S</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>Logical Output Grid</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td>LT(S1, S2)</td>
        <td>Returns 1 if S1 &lt; S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LE(S1, S2)</td>
        <td>Returns 1 if S1 &lt;= S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GT(S1, S2)</td>
        <td>Returns 1 if S1 &gt; S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GE(S1, S2)</td>
        <td>Returns 1 if S1 &gt;= S2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GTLT(S1, S2, S3)</td>
        <td>Returns 1 if S1 &gt; S2 and S1 &lt; S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GELT(S1, S2, S3)</td>
        <td>Returns 1 if S1 &gt;= S2 and S1 &lt; S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GTLE(S1, S2, S3)</td>
        <td>Returns 1 if S1 &gt; S2 and S1 &lt;= S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GELE(S1, S2, S3)</td>
        <td>Returns 1 if S1 &gt;= S2 and S1 &lt;= S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EQ(S1, S2, S3)</td>
        <td>Returns 1 if |S1-S2| &lt;= S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NE(S1, S2, S3)</td>
        <td>Returns 1 if |S1-S2| &gt; S3</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AND(S1, S2, ..., Sn)</td>
        <td>Returns 1 if all Sx &gt; 0</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OR(S1, S2, ..., Sn)</td>
        <td>Returns 1 if at least one Sx &gt; 0</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EOR(S1, S2, ..., Sn)</td>
        <td>Returns 1 if exactly one Sx &gt; 0</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NOT(S)</td>
        <td>Returns 1 if S ==0</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>

.. raw:: html

    <h2>Ensemble Functions</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td>ENS_SAVG(input_arg1 & input_arg2)</td>
        <td>Compute the average of a scalar diagnostic field over an ensemble.  This average is an equally weighted ensemble mean unless weights have been specified in the GDFILE entry as described above or in input_arg2. Input_arg1 specifies the scalar field to average and is a dynamic argument.  Input_arg2 specifies the name of a weight grid to be used in the calculation.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SSPRD(input_arg1)</td>
        <td>Compute the spread (sample standard deviation)  of a scalar diagnostic field over an ensemble.  Input_arg1 specifies the scalar field and is a dynamic argument.  The formula used to compute the spread is SQRT { [ SUM ( Xi**2 ) - ( SUM (Xi) ** 2 ) / N ] / ( N -1 ) } where SUM is the summation over the members, Xi are values of the scalar field from individual ensemble members, and N is                 the number of members.  Note that the average of the squared deviations from the mean is computed by dividing the sum of the squared deviations by ( N - 1 ).</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SSUM(input_arg1)</td>
        <td>Compute the non-weighted sum of the values of a scalar field over an ensemble.  Note that this returns the number of members in the ensemble if "input_arg1"" is a  logical function that always evaluates to 1 [e.g. gt(2<1)]</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SWSPRD(input_arg1 & input_arg2)</td>
        <td>Compute the spred, similar to ENS_SSPRD, of a scalar diagnostic field over an ensemble. The spread is weighted and input_arg2 specifies the name of a weight grid to be used in the calculation.                 NOTE: For uniform weights ens_ssprd (input_arg1) might be expected to give the same result as ens_swsprd(input_arg1 &amp; w1) where  w1 is uniform field of 1.  This does not happen because of the division by (N-1) in ens_ssprd.  The same is also true in comparing ens_vsprd and ens_vwsprd results.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_VAVG(input_arg1 & input_arg2)</td>
        <td>Compute the average of a vector diagnostic field over an ensemble.  This average is an equally weighted ensemble mean unless weights have been specified in the GDFILE entry as described above or in input_arg2. Input_arg1 specifies the vector field to average and is a dynamic argument.  Input_arg2 specifies the name of a weight grid to be used in the calculation.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENC_VSPRD(input_arg1)</td>
        <td>Compute the spread (sample standard deviation) of a vector diagnostic field over an ensemble.  Input_arg1 specifies the vector field and is a dynamic argument.  The formula used to compute the variance for each individual component of a vector is the same as for ENS_SSPRD.  The variances of the components are added, then the square root is taken to compute the vector spread, which is a scalar.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_VWSPRD(input_arg1 & input_arg2)</td>
        <td>Compute the spred, similar to ENS_VSPRD, of a vector diagnostic field over an ensemble. The spread is weighted and input_arg2 specifies the name of a weight grid to be used in the calculation.  Also, see NOTE for function ENS_SWSPRD.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SMAX(input_arg1)</td>
        <td>Compute the maximum of a scalar diagnostic field over an ensemble.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SMIN(input_arg1)</td>
        <td>Compute the minimum of a scalar diagnostic field over an ensemble.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_SRNG(input_arg1)</td>
        <td>Compute the range of a scalar diagnostic field over an ensemble. The range is defined as the difference between the maximum and the minimum.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_PRCNTL(input_arg1 & input_arg2 & input_arg3)</td>
        <td>At each grid point determine the value of input_arg1 such that it exceeds that found in P% of the members of the ensemble, where P is given by input_arg2. The
            first argument is a dynamic argument, the second is a static argument. Input_arg3 specifies the name of a weight grid to be used in the calculation.
            The value of P must range between 0 and 100, and it may vary from grid point to grid point.  If the ensemble members are weighted, it is the sum of the weights associated with the
            order statistics that determines the value corresponding to the percentile.  If the percentile falls between two members, the value is determined by interpolation</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_MODE(input_arg1 & input_arg2)</td>
        <td>At each grid point, determine the mode of a weighted ensemble distribution.  The mode value is the first one
            found by a trisection algorithm.  The algorithm uses a moving, shrinking data interval in which the weight sums for the data
            values are made in each of three subintervals.  The largest weight sum determines the next interval, with ties decided in
            the direction of the data mean.  The algorithm terminates when only the middle subinterval has a non-zero sum of weights.  The
            weighted average of the data values in this lone subinterval is the mode.  Input_arg2 specifies the name of a weight
            grid to be used in the calculation. </td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_PROB(input_arg1)</td>
        <td>Compute the probability of the multi-variate expression given in the input argument.  The expression in the input argument is composed of the logical functions AND, OR, and EOR operating on the logical arithmetic comparison functions LT, LE, GT, GE, GTLT, GELT, GTLE, and GELE.  ENS_PROB computes the weighted average of the 1 or 0 logical evaluations over the members of the ensemble to yield the relative frequency, or probability, of occurrence.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_CPRB(input_arg1 & input_arg2 & input_arg3 & input_arg4)</td>
        <td>At each grid point, determine the univariate cumulative probability that the value of a given function is less than or equal to a given value in an ideal ensemble.  The input arguments are, respectively, the function to evaluate, the given value whose cumulative probability is sought, the lower bound below which values of the function are impossible, and the upper bound above which values of the function are impossible.  If a value for input_arg2 is outside of a given bound, the result is a missing value.  The first argument is a dynamic argument; the others are static.  The last two are optional.  To omit input_arg3 but include input_arg4, specify input as ( input_arg1 &amp; input_arg2 &amp; &amp; input_arg4 ).All arguments may be functions that vary from one grid point to another.  For an ideal ensemble, there is a 2/(N+1) probability of values falling outside of the ensemble envelope.  The probability density function is piecewise linear with triangular extrapolations beyond the ensemble, limited by any bounding values given.  The following example finds the cumulative probability o</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENS_CVAL(input_arg1 & input_arg2 & input_arg3 & input_arg4)</td>
        <td>At each grid point, determine the value of a given function such that the univariate cumulative probability of values less than or equal to the computed value is  the given probability, assuming an ideal ensemble.  The input arguments are, respectively, the function to evaluate, the given cumulative probability ranging from 0 to 1, the lower bound below which values of the function are impossible, and the upper bound above which values of the function are  impossible.  A computed value will never fall outside of a bounding value.  The first argument is a dynamic argument; the others are static.  The last two are optional.  To omit input_arg3 but include input_arg4, specify input as  ( input_arg1 &amp; input_arg2 &amp; &amp; input_arg4 ).  All arguments may be functions that vary from one grid point to another.  If the value of the second argument is not between 0 and 1, inclusive, then a missing value is assigned at that grid point. For an ideal ensemble, there is a 2/(N+1) probability of values falling outside of the ensemble envelope.  The probability  density function is piecewise linear with triangular extrapolations beyond the ensemble, limited by any bounding values given.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>Layer Functions</h2>
    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td>LYR_SWTM(input_arg1 | levels)</td>
        <td>Computes the weighted mean of a scalar field.  If the input argument is a single-level parameter  or function, the weight is the depth of the layer extending from point halfway between the level and the next lower level and the point halfway between the level and the next upper level. If the input argument is a two-level function, then the weight is the depth of the two-level layer used in calculating the function. In isobaric coordinates, the logarithm of pressure is used to compute the depths.  Input_arg1 specifies diagnostic function to average over the layers and output_arg is not required.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LYR_MXMN(argin & fldout [&argout] [|lvls]</td>
        <td>Computes the following over the specified range of levels...  1.The maximum or minimum value of a scalar quantity. 2.The value of a second output function coincident with the extrema of the input function.  It might be a level, such as HGHT or PRES. It might be some other function, e.g. absolute vorticity.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LYR_FVONISFC(fvalu & fisfc & visfc & n & gradflag [|lvls]</td>
        <td>Computes the value of a function on a specified isosurface of a second function. The function  will have the ability to  traverse the atmosphere starting from the bottom level upwards or top level           downwards and return either the 1st, 2nd, or nth level at which the isosurface threshold value exist.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>GUI Programs</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>Program</th>
        <th>Description</th>
        <th>Python/MetPy Equivalent</th>
      </tr>
      <tr>
        <td>nfax</td>
        <td>Displays NMX 6 bit raster fax products</td>
        <td></td>
      </tr>
      <tr>
        <td>nmap</td>
        <td>Displays and animates different types of meteorological data on a geographic background.</td>
        <td></td>
      </tr>
      <tr>
        <td>nsharp</td>
        <td>Advanced interactive sounding analysis program</td>
        <td></td>
      </tr>
      <tr>
        <td>ntrans</td>
        <td>Displays and animates N-AWIPS graphics metafiles in an X-Window.</td>
        <td></td>
      </tr>
      <tr>
        <td>nwx</td>
        <td>Displays text products from the Family of Services (FOS) data feed.</td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>Decoders</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td>dcacars</td>
        <td>NetCDF ACARS data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcacft</td>
        <td>Raw AIREP, PIREP, RECCO, and AMDAR reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcairm</td>
        <td>AIRMET (AIRman's METeorological Information) reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dccosmic</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dccsig</td>
        <td>Convective signet and convective outlook reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcffa</td>
        <td>Flash flood watch reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcffg</td>
        <td>Flash flood guidance data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcgmos</td>
        <td>GFS MOS</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcgrib</td>
        <td>GRIB</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcgrib2</td>
        <td>GRIB2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dchrcn</td>
        <td>Forecast/advisory reports for tropical depressions, tropical storms, and hurricanes for the Atlantic and Pacific oceans.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcigdr</td>
        <td>IGDR data in BUFR format</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcisig</td>
        <td>SIGMET reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dclsfc</td>
        <td>Land surface synoptic reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcmetr</td>
        <td>Raw SAO and METAR reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcmsfc</td>
        <td>Raw buoy, ship, C-MAN, and Coast Guard reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcncon</td>
        <td>Non-convective SIGMET reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcncprof</td>
        <td>NetCDF format profiler and RASS reports from NOAA/FSL</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcnexr2</td>
        <td>CRAFT IDD data stream</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcnldn</td>
        <td>NLDN lightning data reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcnmos</td>
        <td>NGM MOS</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcprof</td>
        <td>BUFR format profiler reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcrdf</td>
        <td>Regional digital forecast (RDF)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcredbook</td>
        <td>Creates displays of Redbook graphic format products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcscd</td>
        <td>Supplemental Climatological Data reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcshef</td>
        <td>Raw SHEF reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcshef_comet</td>
        <td>Data feed (LDM) raw SHEF reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcstorm</td>
        <td>Severe storm reports from SPC</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcsuomi</td>
        <td>NetCDF format SUOMINET data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcsvrl</td>
        <td>Severe local storm reports (tornado and severe thunderstorm watch reports)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dctaf</td>
        <td>Raw terminal aerodrome forecast (TAF) reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dctama</td>
        <td>TAMDAR data in BUFR format</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dctrop</td>
        <td>Hurricane/tropical storm reprots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcuair</td>
        <td>Upper air sounding data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcuspln</td>
        <td>USPLN lightning data reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwarn</td>
        <td>Flash flood, tornado, and severe thunderstorm warning reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwatch</td>
        <td>WWUS40 format severe thunderstorm and tornado watch box bulletins</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwcn</td>
        <td>Watch county notification reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwcp</td>
        <td>Tornado and severe thunderstorm Watch Corner Points (WCP) reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwou</td>
        <td>Watch outline update reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwstm</td>
        <td>Winter storm reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcwtch</td>
        <td>Tornado and severe thunderstorm watch box reports and watch status reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>dcxmos</td>
        <td>GFSX MOS</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>Programs</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK Function</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td>acprof</td>
        <td>Draws profiles of ACARS data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>bfr2gp</td>
        <td>Transfers data from a Jack Woollen BUFR file into GEMPAK sounding and surface data files</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>bufrenc</td>
        <td>Processes ASCII input file to produce one or more BUFR output files</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>cpcgstn</td>
        <td>Searches for stations located inside specified area from a vector graphics file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gd2ndfd</td>
        <td>Converts a GEMPAK grid to an NDFD GRIB2 file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdbiint</td>
        <td>Interpolates grids from one projection to another</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdcfil</td>
        <td>Creates a GEMPAK grid file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">gdcntr</td>
        <td class="tg-implemented">Draws contour lines through a scalar grid</td>
        <td class="tg-implemented"><a href="https://matplotlib.org/devdocs/api/_as_gen/matplotlib.axes.Axes.contour.html">matplotlib.axes.Axes.contour</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td>gdcross</td>
        <td>Displays a vertical cross section of scalar and/or vector grids</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gddelt</td>
        <td>Deletes grids from GEMPAK grid files</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gddiag</td>
        <td>Computes a scalar/vector diagnostic grid and adds it to the grid file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdedit</td>
        <td>Reads grids from a sequential edit file and adds them to a GEMPAK grid file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdfrzl</td>
        <td>Generates GFA FZLVLs in VG format from a scalar grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdgrib2</td>
        <td>Computes a scalar diagnostic grid and adds it to a GRIB2 file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdgrib</td>
        <td>Computes a scalar diagnostic grid and adds it to a GRIB file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdgsfc</td>
        <td>Computes grid data and interpolates to stations in a GEMPAK surface file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdinfo</td>
        <td>Lists information about GEMPAK grid files</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdlist</td>
        <td>Lists data from a scalar grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdmap</td>
        <td>Plots data from a scalar grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdmod</td>
        <td>Moves grids from one GEMPAK grid file to another</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdobscnt</td>
        <td>Creates a gridded sampling of the number of surface observations within a specified radius of each grid point.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdomeg</td>
        <td>Computes grids of vertical motion and adds them to the grid file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdplot2</td>
        <td>Draws contour lines through scalar grids and/or wind barbs or arrows or streamlines through vector grids. Multiple sets of contours, vectors, and/or streamlines can be generated for each frame.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdplot3</td>
        <td>Draws contour lines through scalar grids and/or wind barbs or arrows or streamlines through vector grids. the program also plots contents of a text file and/or objects. Multiple sets of contours, vectors, streamlines, objects and/or text files can be generated for each frame.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdplot</td>
        <td>Draws contour lines through scalar grids and/or wind barbs or arrows through vector grids. Multiple sets of contours and vectors can be generated for each frame.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdprof</td>
        <td>Draws profiles of a scalar grid and/or a vector grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdptpdf</td>
        <td>Raw cumulative probability or probability density at selected point and time at a single level.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdpvsf</td>
        <td>Vertically interpolates grid data to an arbitrary surface.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdradrc</td>
        <td>Creates a gridded composite of Canadian ASCII products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdradr</td>
        <td>Creates a gridded composite of NEXRAD level III products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdstat</td>
        <td>Computes statistics on a time series of scalar grids</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdstream</td>
        <td>Draws streamlines through a vector grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdthgt</td>
        <td>Draws contours and wind barbs or arrows on a time selection at a point within a grid</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdtopo</td>
        <td>Creates a GEMPAK GRID file from a raster file of topography or land use values.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdtser</td>
        <td>Draws a time series of a scalar at a single level</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdtspdf</td>
        <td>Draws contour or color fill of cumulative probability or probability density as a function of grid values and forecast time at a selected point and level or layer.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdvint</td>
        <td>Performs interpolation between vertical coordinates</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gdwind</td>
        <td>Displays a vector grid using wind barbs or arrows</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpanot</td>
        <td>Will allow the user to place objects at any location on the current graphic device</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpbox</td>
        <td>Draws a box around a region</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpclear</td>
        <td>Clears the current graphics device</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpclose</td>
        <td>Closes the specified graphics output window or file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpcolor</td>
        <td>Changes the colors on a color device</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpend</td>
        <td>Terminates the GEMPLT subprocesses.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpfax</td>
        <td>Creates a postscript, GIF, or TIFF file, or an X Windows display from a 6-bit fax file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpfront</td>
        <td>Version of GPMAP that plots map symbols interpreted from ASUS1 bulletins. Forecast front positions from FSUS2 bulletins can be plotted by specifying the forecast hour desired.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpltln</td>
        <td>Draws a map, LAT/LON lines with a selected marker, and various image and graphic products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpmap</td>
        <td>Draws a map, LAT/LON lines, and various images and graphic products.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpnexr2</td>
        <td>Displays NEXRAD level II products.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpnids</td>
        <td>Plots NEXRAD level III products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gprcm</td>
        <td>Version of GPMAP that plots Radar Coded Message (RCM) bulletins</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpscat</td>
        <td>Draws scatterometer wave height and wind maps, LAT/LON lines, and various controllable image and graphic customizations.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gptcww</td>
        <td>Generates the track error watch/warn pilot for the TPC, using breakpoint information contained in a VGF file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gptext</td>
        <td>Draws the contents of a text file to the output device.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gptpc</td>
        <td>Generates four hurricane graphics from the TPC. 1) Wind swath plot 2) Strike probability plot 3) Wind intensity graph 4) Wind speed table</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gptraj</td>
        <td>Computes trajectories for gridded data files</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpvad</td>
        <td>Plots NEXRAD level III VAD wind profile</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>gpwarn</td>
        <td>Version of GPMAP that plots filled county/zone regions from reports which use the universal generic county/zone identified lines.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>grphgd</td>
        <td>Generates a grid either from contours drawn via NMAP product generation, or from a provided VGF file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>img2gd</td>
        <td>Converts an image to a grid. The grid navigation specified may be different than that of the source image.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>mkelev</td>
        <td>NOT FOUND</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nagrib2</td>
        <td>Converts gridded data in GRIB2 files to GEMPAK gridded data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nagrib</td>
        <td>Converts gridded data in GRIB files to GEMPAK gridded data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>namsnd</td>
        <td>Transfers model profile output in BUFR to GEMPAK sounding and surface data files.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ndfdg2</td>
        <td>Converts NDFD gridded data in GRIB2 files to GEMPAK gridded data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nex2gini</td>
        <td>Creates as GINI format image composite of NEXRAD level III products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nex2img</td>
        <td>Creates a GIF format image composite of NEXRAD level III products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nexr2rhi</td>
        <td>Displays NEXRAD level II vertical cross sections</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>nexrcomp</td>
        <td>Creates a GINI format image composite of NEXRAD level III products</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>oabox</td>
        <td>Draws a box around an objective analysis region</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>oabsfc</td>
        <td>Performs a Barnes objective analysis on surface data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>oabsnd</td>
        <td>Performs a Barnes objective analysis on upper air data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>oagrid</td>
        <td>Creates a GEMPAK grid file which can be used in a Barnes objective analysis program</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sector</td>
        <td>Subsets a GINI satellite image in area and pixel resolution.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfcfil</td>
        <td>Creates a new GEMPAK surface file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfchck</td>
        <td>Reads a GEMPAK surface data file and produces a table of stations and an indicator showing whether each station reported data at each time in the file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfcntr</td>
        <td>Plots surface station data on a map and optionally contours one of the fields being plotted</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfdelt</td>
        <td>Deletes data from a surface data file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfdsl604</td>
        <td>Lists data from a GEMPAK surface file in a fixed format for use on the AMS DataStreme website</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfedit</td>
        <td>Adds or changes data in a surface file using a sequential text file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfgram</td>
        <td>Draws a meteorogram for surface data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfl604</td>
        <td>Lists data from a GEMPAK surface file in a fixed format</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sflist</td>
        <td>Surface data from a GEMPAK surface data file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfmap</td>
        <td>Plots surface station data on a map</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfmod</td>
        <td>Moves selected surface data from an input surface file to an output surface file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfstns</td>
        <td>Modifies the station information in a surface file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sfvgsf</td>
        <td>Adds or changes data in a surface file using the elements found in a Vector Graphics File</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sncfil</td>
        <td>Creates a new GEMPAK sounding file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sncross</td>
        <td>Draws cross sections through sounding data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sndelt</td>
        <td>Deletes data from a sounding data file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sndslist</td>
        <td>Lists upper air data from a sounding file for specified vertical levels and stations in a format used for the AMS DataStreme web site.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snedit</td>
        <td>Adds data in a sequential edit file to a sounding file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snhodo</td>
        <td>Draws a hodograph of upper air data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snlist</td>
        <td>Lists upper air data from a sounding file for specified vertical levels and stations</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snmap</td>
        <td>Plots sounding data on a map</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snmod</td>
        <td>Moves selected sounding data from an input sounding file to an output sounding file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snprof</td>
        <td>Draws profiles of upper air data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>snstns</td>
        <td>Modifies the station information in the upper air file</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>sntser</td>
        <td>Draws a time series at a sounding station</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>vg2uka</td>
        <td>Converts VG files to ASCII files, using the UKMET browsable ASCII format.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>


.. raw:: html

    <h2>Parameters</h2>

    <table class="wy-table-responsive">
      <tr>
        <th>GEMPAK</th>
        <th>Description</th>
        <th>Python/MetPy Equilvalent</th>
        <th>Grid Compatible?</th>
        <th>Tested against GEMPAK?</th>
        <th>Units?</th>
      </tr>
      <tr>
        <td><b>Temperature Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TMPC</td>
        <td>Temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TMPF</td>
        <td>Temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TMPK</td>
        <td>Temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STHA</td>
        <td>Surface potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STHK</td>
        <td>Surface potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STHC</td>
        <td>Surface potential temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STHE</td>
        <td>Surface equivalent potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STHS</td>
        <td>Surface saturation equivalent potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTA</td>
        <td>Potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTK</td>
        <td>Potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTC</td>
        <td>Potential temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTE</td>
        <td>Equivalent potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTS</td>
        <td>Saturation equivalent potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVRK</td>
        <td>Virtual temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVRC</td>
        <td>Virtual temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVRF</td>
        <td>Virtual temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTV</td>
        <td>Virtual potential temperature in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDXC</td>
        <td>Maximum 24 hour temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDNC</td>
        <td>Minimum 24 hour temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDXF</td>
        <td>Maximum 24 hour temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDNF</td>
        <td>Minimum 24 hour temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>T6XC</td>
        <td>Maximum 6 hour temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>T6NC</td>
        <td>Minimum 6 hour temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>T6XF</td>
        <td>Maximum 6 hour temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>T6NF</td>
        <td>Minimum 6 hour temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DMAX</td>
        <td>Daily weather map maximum temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DMIN</td>
        <td>Daily weather map minimum temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SSTC</td>
        <td>Sea surface temperature in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SSTF</td>
        <td>Sea surface temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LTMP</td>
        <td>Temperature in Celsius of surface air lifted to 500 or !x mb</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Moisture Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DWPC</td>
        <td>Dew point in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DWPF</td>
        <td>Dew point in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DWPK</td>
        <td>Dew point in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DPDC</td>
        <td>Dew point depression in Celsius</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DPDF</td>
        <td>Dew point depress in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DPDK</td>
        <td>Dew point depression in Kelvin</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MIXR</td>
        <td class="tg-implemented">Mixing ratio</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.mixing_ratio.html#metpy.calc.mixing_ratio">metpy.calc.mixing_ratio</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">MIXS</td>
        <td class="tg-implemented">Saturated mixing ratio</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.saturation_mixing_ratio.html#metpy.calc.saturation_mixing_ratio">metpy.calc.saturation_mixing_ratio</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMXR</td>
        <td>Surface mixing ratio</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMXS</td>
        <td>Surface saturated mixing ratio</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">RELH</td>
        <td class="tg-implemented">Relative humidity</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.relative_humidity_from_dewpoint.html#metpy.calc.relative_humidity_from_dewpoint">metpy.calc.relative_humidity_from_dewpoint</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TMWK</td>
        <td class="tg-implemented">Wet bulb temperature in Kelvin</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.wet_bulb_temperature.html#metpy.calc.wet_bulb_temperature">metpy.calc.wet_bulb_temperature(pressures, temperatures, dewpoints).to('K')</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TMWC</td>
        <td class="tg-implemented">Wet bulb temperature in Celsius</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.wet_bulb_temperature.html#metpy.calc.wet_bulb_temperature">metpy.calc.wet_bulb_temperature(pressures, temperatures, dewpoints).to('degC')</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TMWF</td>
        <td class="tg-implemented">Wet bulb temperature in Fahrenheit</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.wet_bulb_temperature.html#metpy.calc.wet_bulb_temperature">metpy.calc.wet_bulb_temperature(pressures, temperatures, dewpoints).to('degF')</a></td>
        <td></td>
        <td class="tg-no">No</td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">VAPR</td>
        <td class="tg-implemented">Vapor pressure in millibars</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.vapor_pressure.html#metpy.calc.vapor_pressure">metpy.calc.vapor_pressure</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">VAPS</td>
        <td class="tg-implemented">Saturation vapor pressure in millibars</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.saturation_vapor_pressure.html#metpy.calc.saturation_vapor_pressure">metpy.calc.saturation_vapor_pressure</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">LHVP</td>
        <td class="tg-implemented">Latent heat of vaporization</td>
        <td class="tg-implemented"><a href="api/generated/metpy.constants.html">metpy.constants.water_heat_vaporization</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-info">PWTR</td>
        <td class="tg-info">Precipitable water at a given level</td>
        <td class="tg-info"><a href="api/generated/metpy.calc.precipitable_water.html#metpy.calc.precipitable_water">metpy.calc.precipitable_water</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Height Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGHT</td>
        <td>Height in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGTM</td>
        <td>Height in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGTK</td>
        <td>Height in kilometers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGTD</td>
        <td>Height in decameters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGFT</td>
        <td>Height in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGFH</td>
        <td>Height in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGFK</td>
        <td>Height in thousands of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HGML</td>
        <td>Height in miles</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DHGT</td>
        <td>Dry hydrostatic height in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MHGT</td>
        <td>Moist hydrostatic height in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STDZ</td>
        <td>Character standard height convention used on ua charts</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RSTZ</td>
        <td>Numeric standard height convention used on ua charts</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ZMSL</td>
        <td>Estimated height at a pressure level</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Pressure and Altimeter Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PRES</td>
        <td>Station pressure in millibars</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PMSL</td>
        <td>Mean sea level pressure</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PALT</td>
        <td>Surface pressure in millibars from ALTI</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ALTI</td>
        <td>Altimeter setting in inches of mercury</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ALTM</td>
        <td>Altimeter setting converted to millibars</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SALT</td>
        <td>Abbreviated standard altimeter setting</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMSL</td>
        <td>Abbreviated mean sea level pressure in millibars</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SALI</td>
        <td>Abbreviated altimeter setting in inches of mercury</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RMSL</td>
        <td>First 3 digits left of decimal of PMSL * 10</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RSLI</td>
        <td>First 3 digits left of decimal of ALTI * 100</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RSLT</td>
        <td>First 3 digits let of decimal of ALTM * 10</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PTND</td>
        <td>Pressure tendency</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PTSY</td>
        <td>Graphics symbol for pressure tendency</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>P03C</td>
        <td>3-h numeric pressure change</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>P03D</td>
        <td>Pressure tendency and change group, appp</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>P24C</td>
        <td>2y-h numeric pressure change</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PANY</td>
        <td>Returns PMSL if available, if not, returns ALTM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RANY</td>
        <td>Computes the 3 digit display of pressure</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SANY</td>
        <td>Creates a 3 character string from integral part of PMSL or ALTM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Winds</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>UWND</td>
        <td>u wind in m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VWND</td>
        <td>V wind in m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>UKNT</td>
        <td>U wind in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VKNT</td>
        <td>V wind in knows</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DRCT</td>
        <td>Wind direction in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SPED</td>
        <td>Wind speed m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKNT</td>
        <td>Wind speed knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SMPH</td>
        <td>Wind speed mph</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PSPD</td>
        <td>Packed direction and speed m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PKNT</td>
        <td>Packed direction and speed in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GUST</td>
        <td>Wind gusts in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GUMS</td>
        <td>Wind gusts in m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWDR</td>
        <td>Peak 5 -second wind direction in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWSP</td>
        <td>Peak 5-second wind speed in m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWHR</td>
        <td>Hour of 5-second peak wind</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWMN</td>
        <td>Minutes of 5-second peak wind</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WNML</td>
        <td>Wind component toward a direction 90 degrees CCW from specified direction</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCMP</td>
        <td>Wind component toward a specified direction</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BARB</td>
        <td>Barb m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BRBM</td>
        <td>Barb m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BRBK</td>
        <td>Barb knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BRBS</td>
        <td>Barb mi/hr</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ARRW</td>
        <td>Arrows m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ARRM</td>
        <td>Arrows m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ARRK</td>
        <td>Arrows knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DARR</td>
        <td>Wind direction arrows of uniform length</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Lifted Condenstaion Level</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">TLCL</td>
        <td class="tg-implemented">Temperature in Kelvin at the LCL from the given level</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html#metpy.calc.lcl">metpy.calc.lcl</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">PLCL</td>
        <td class="tg-implemented">Pressure in millibar at the LCL from the given level</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html#metpy.calc.lcl">metpy.calc.lcl</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Stability Indices</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">BRCH</td>
        <td class="tg-notimplemented">Bulk Richardson number</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/628">Issue #628</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">BRCV</td>
        <td class="tg-notimplemented">BRCH computed using CAPV</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">BVFQ</td>
        <td class="tg-implemented">Brunt-Vaisala frequency in a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.brunt_vaisala_frequency.html#metpy.calc.brunt_vaisala_frequency">metpy.calc.brunt_vaisala_frequency</a> (use with <a href="api/generated/metpy.calc.mixed_layer.html#metpy.calc.mixed_layer">metpy.calc.mixed_layer</a> to obtain layer average)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">BVPD</td>
        <td class="tg-implemented">Brunt-Vaisala period in a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.brunt_vaisala_period.html#metpy.calc.brunt_vaisala_period">metpy.calc.brunt_vaisala_period</a> (use with <a href="api/generated/metpy.calc.mixed_layer.html#metpy.calc.mixed_layer">metpy.calc.mixed_layer</a> to obtain layer average)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">BVSQ</td>
        <td class="tg-implemented">Brunt-Vaisala frequency squared in a layer</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.brunt_vaisala_frequency_squared.html#metpy.calc.brunt_vaisala_frequency_squared">metpy.calc.brunt_vaisala_frequency_squared</a> (use with <a href="api/generated/metpy.calc.mixed_layer.html#metpy.calc.mixed_layer">metpy.calc.mixed_layer</a> to obtain layer average)</td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">CAPE</td>
        <td class="tg-implemented">Convective available potential energy</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.cape_cin.html#metpy.calc.cape_cin">metpy.calc.cape_cin</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">CAPV</td>
        <td class="tg-notimplemented">CAPE computed by using virtual temperature</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">CINS</td>
        <td class="tg-implemented">Convective inhibition</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.cape_cin.html#metpy.calc.cape_cin">metpy.calc.cape_cin</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">CINV</td>
        <td class="tg-notimplemented">CINS computed using virtual temperature</td>
        <td class="tg-notimplemented"></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">CTOT</td>
        <td class="tg-notimplemented">cross totals index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/633">Issue #633</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">EQLV</td>
        <td class="tg-implemented">equilibrium level</td>
        <td class="tg-implemented"<a href="api/generated/metpy.calc.el.html#metpy.calc.el">metpy.calc.el</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EQTV</td>
        <td>EQLV computed using virtual temperature</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">KINX</td>
        <td class="tg-notimplemented">K index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/631">Issue #631</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">LAPS</td>
        <td class="tg-notimplemented">temperature lapse rate in a layer</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/638">Issue #638</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">LCLP</td>
        <td class="tg-implemented">pressure in millibars at the LCL from surface</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html#metpy.calc.lcl">metpy.calc.lcl</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">LCLT</td>
        <td class="tg-implemented">temperature in Kelvin at the lcl from the surface</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lcl.html#metpy.calc.lcl">metpy.calc.lcl</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">LFCT</td>
        <td class="tg-implemented">level of free convection by comparing temperature between a parcel and the environment</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.lfc.html#metpy.calc.lfc">metpy.calc.lfc</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LFCV</td>
        <td>LFCT computed using the virtual temperature</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">LIFT</td>
        <td class="tg-notimplemented">lifted index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/632">Issue #632</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">LFTV</td>
        <td class="tg-notimplemented">LIFT computed using the virtual temperature</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/632">Issue #632</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">LHAN</td>
        <td class="tg-notimplemented">low elevation Haines index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/635">Issue #635</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">MHAN</td>
        <td class="tg-notimplemented">middle elevation Haines index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/635">Issue #635</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">HHAN</td>
        <td class="tg-notimplemented">high elevation Haines index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/635">Issue #635</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MLMR</td>
        <td>mean mixed layer MIXR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MLTH</td>
        <td>mean mixed layer THTA</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">PWAT</td>
        <td class="tg-implemented">Precipitable water for the entire sounding</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.precipitable_water.html#metpy.calc.precipitable_water">metpy.calc.precipitable_water</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">RICH</td>
        <td class="tg-notimplemented">Richardson number in a layer</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/647">Issue #647</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SEPA</td>
        <td>Isentropic pressure thickness in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">SHOW</td>
        <td class="tg-notimplemented">Showalter index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/630">Issue #630</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHRD</td>
        <td>wind shear direction in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHRM</td>
        <td>wind shear magnitude in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STAB</td>
        <td>THTA lapse rate in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STAP</td>
        <td>THTA change with pressure in a layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">SWET</td>
        <td class="tg-notimplemented">SWEAT index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/634">Issue #634</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">TOTL</td>
        <td class="tg-notimplemented">Total totals index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/633">Issue #633</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-notimplemented">VTOT</td>
        <td class="tg-notimplemented">vertical totals index</td>
        <td class="tg-notimplemented"><a href="https://github.com/Unidata/MetPy/issues/633">Issue #633</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Cloud Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>xCLD</td>
        <td>Character cloud coverage code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCLD</td>
        <td>xCLD at maximum cloud coverage</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>xCLO</td>
        <td>Fractional cloud coverage</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCLO</td>
        <td>xCLO at maximum cloud coverage</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLCx</td>
        <td>Numeric cloud coverage</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLCT</td>
        <td>CLCx at maximum cloud coverage</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLDS</td>
        <td>combined cloud coverage short code from three levels</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CMBC</td>
        <td>combined cloud coverage numeric from three levels</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLHx</td>
        <td>cloud height in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLDx</td>
        <td>combined cloud height and short code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLDT</td>
        <td>CLDx at maximum coverage level</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLDB</td>
        <td>CLDx at the lowest ceiling/layer</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>COMx</td>
        <td>numeric combined cloud height and coverage combined</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>COMT</td>
        <td>COMx at maximum coverage level</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CHCx</td>
        <td>Numeric combined cloud height and coverage combined</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CHDx</td>
        <td>combined cloud height and short code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CEIL</td>
        <td>ceiling in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CFRL</td>
        <td>fraction of celestial dome covered by all low and mid level clouds from WMO code 2700</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTYL</td>
        <td>low-level cloud genera from WMO code 0513</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTYM</td>
        <td>mid-level cloud genera from WMO code 0515</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTYH</td>
        <td>high-level cloud genera from WMO code 0509</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CBAS</td>
        <td>cloud base height from WMO code 1600</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CSYL</td>
        <td>cloud graphics symbol for CTYL</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CSYM</td>
        <td>cloud graphics symbol for CTYM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CSYH</td>
        <td>cloud graphics symbol for CTYH</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CSYT</td>
        <td>cloud graphics symbol for first level reporting clouds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CFRT</td>
        <td>cloud coverage number from CLCT (maximum clouds)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKYC</td>
        <td>cloud coverage graphics symbol for CFRT</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKYM</td>
        <td>sky coverage symbol with wind barbs m/s</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKYK</td>
        <td>sky coverage symbol with wind barbs knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>XVFR</td>
        <td>categorical identification of flight rules</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Weather Codes</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCOD</td>
        <td>character weather code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WNUM</td>
        <td>numeric weather code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WTMO</td>
        <td>character WMO weather code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WWMO</td>
        <td>numeric WMO weather code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSYM</td>
        <td>graphics weather symbol corresponding to WWMO</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWTH</td>
        <td>character past weather WMO code or graphics symbol for it</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PWWM</td>
        <td>numeric past weather WMO code</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Station Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STID</td>
        <td>character station identifier</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNM</td>
        <td>station number</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SLAT</td>
        <td>station latitiude in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SLON</td>
        <td>station longitude in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SELV</td>
        <td>station elevation in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RANG</td>
        <td>range in kilometers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AZIM</td>
        <td>azimuth in kilometers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LATI</td>
        <td>latitude in degrees from range/azimuth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LONG</td>
        <td>longitude in degrees from range/azimuth</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DELT</td>
        <td>delta time in seconds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Model Output Statistics</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MXMN</td>
        <td>maximum or minimum temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TNTF</td>
        <td>night temperature fcst in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TNCF</td>
        <td>night temperature climatology in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TNAF</td>
        <td>night temperature anomaly in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDYF</td>
        <td>day temperature fcst in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDCF</td>
        <td>fay temperature climatology in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDAF</td>
        <td>day temperature anomaly in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CL12</td>
        <td>prevailing total sky cover fcst for a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SK12</td>
        <td>maximum sustain surface wind spped fcst for 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP06</td>
        <td>probability of precip fcst in 6-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP12</td>
        <td>probability of precip fcst in 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP1C</td>
        <td>probability of precipitation climatology in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP1A</td>
        <td>probability of precipitation anomaly in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP24</td>
        <td>probability of precipitation fcsr in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP2C</td>
        <td>probability of precipitation climatology in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PP2A</td>
        <td>probability of precipitation anomaly in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QP06</td>
        <td>Quantitative precipitation fcst in a 6-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QPX2</td>
        <td>Maximum amount of precipitation in inches fcst in a 12-hr period.  Values are same as QP12.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QP12</td>
        <td>Quantitative precipitation fcst in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QP24</td>
        <td>Quantitative precipitation fcst in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TS06</td>
        <td>Unconditional probability of thunderstorms occurring in a 6-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TS12</td>
        <td>Unconditional probability of thunderstorms occurring in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TS24</td>
        <td>Unconditional probability of thunderstorms occurring in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TC06</td>
        <td>Unconditional probability of severe weather occurring in a 6-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TC12</td>
        <td>Unconditional probability of severe weather occurring in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PCPT</td>
        <td>Categorical forecast of precipitation</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POZP</td>
        <td>Conditional probability of freezing precipitation</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POSN</td>
        <td>Conditional probability of snow</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SN06</td>
        <td>Categorical forecast of snow amount falling in a 6-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SN12</td>
        <td>Categorical forecast of snow amount falling in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SN24</td>
        <td>Categorical forecast of snow amount falling in a 24-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PZ12</td>
        <td>Conditional probability of freezing precipitation in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PS12</td>
        <td>Conditional probability of snow in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PR12</td>
        <td>Conditional probability of mixed liquid/frozen precipitation in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PC12</td>
        <td>Categorical forecast of precipitation type in a 12-hr period</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FCIG</td>
        <td>Categorical forecast of ceiling height conditions</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FVIS</td>
        <td>Categorical forecast of visibility conditions</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FVSA</td>
        <td>Categorical forecast of visibility conditions (for new MOS)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OVIS</td>
        <td>Categorical forecast in plain language of obstructions to vision</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WXPB</td>
        <td>Categorical weather precipitation probability or areal coverage determined by the precipitation parameter having the highest probability or areal coverage in WNUM.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>TAF Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TDRC</td>
        <td>Temporary/probability wind direction in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TSKN</td>
        <td>Temporary/probability wind speed in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TGST</td>
        <td>Temporary/probability wind gusts in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BRGK</td>
        <td>Gust barb feathered in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCHx</td>
        <td>Temporary/probability numeric combined cloud height and coverage, as for CHCx</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCEL</td>
        <td>Temporary/probability ceiling in hundreds of feet, as for CEIL</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TSKC</td>
        <td>Temporary/probability cloud coverage graphics symbol, as for SKYC</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TXVF</td>
        <td>Temporary/probability categorical identification of flight rules, as for XVFR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TWNM</td>
        <td>Temporary/probability numeric weather code, as for WNUM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TWSY</td>
        <td>Temporary/probability graphics weather symbol corresponding to TWNM, as for WSYM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVSB</td>
        <td>Temporary/probability visibility in statute miles</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PPRB</td>
        <td>Probability for TAF forecast change indicator</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VWNM</td>
        <td>Vicinity numeric weather code, as for WNUM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VWSY</td>
        <td>Vicinity graphics weather symbol corresponding to VWNM, as for WSYM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVWN</td>
        <td>Temporary/probability vicinity numeric weather code, as for WNUM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSKC</td>
        <td>Worst case cloud coverage graphics symbol, as for SKYC</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WXVF</td>
        <td>Worst case categorical identification of flight rules, as for XVFR</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TPWN</td>
        <td>Temporary/probability/vicinity numeric weather code, as for WNUM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TPWS</td>
        <td>Temporary/probability/vicinity graphics weather symbol corresponding to TPWN, as for WSYM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AWNM</td>
        <td>Prevailing/temporary/probability/vicinity numeric weather code, as for WNUM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AWSY</td>
        <td>Prevailing/temporary/probability/vicinity graphics weather symbol corresponding to AWNM, as for WSYM</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LLWS</td>
        <td>Low level wind shear forecast flag</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MOTV</td>
        <td>Mountain obscuration threshold value in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CMSL</td>
        <td>Ceiling converted to mean sea level in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MOBS</td>
        <td>Mountain obscuration threshold met indicator</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCMS</td>
        <td>Temporary/probability ceiling converted to mean sea level in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TMOB</td>
        <td>Temporary/probability mountain obscuration threshold met indicator</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCMS</td>
        <td>Worst case ceiling converted to mean sea level in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WMOB</td>
        <td>Worst case mountain obscuration threshold met indicator</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCTL</td>
        <td>Temporary/probability low-level cloud genera from WMO Code 0513, as for CTYL</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCSL</td>
        <td>Temporary/probability cloud graphics symbol for TCTL, as for CSYL</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Marine Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WHGT</td>
        <td>Wave height in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WHFT</td>
        <td>Wave height in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WPER</td>
        <td>Wave period in seconds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HOWW</td>
        <td>Height of wind wave in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POWW</td>
        <td>Period of wind wave in seconds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HOSW</td>
        <td>Height of predominant swell wave in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POSW</td>
        <td>Period of predominant swell wave in seconds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DOSW</td>
        <td>Direction of predominant swell wave in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HOS2</td>
        <td>Height of secondary swell wave in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POS2</td>
        <td>Period of secondary swell wave in seconds</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DOS2</td>
        <td>Direction of secondary swell wave in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAV2</td>
        <td>Combined wind wave period and height in feet ("2 group")</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAV3</td>
        <td>Combined predominant and secondary swell wave direction in tens of degrees ("3 group")</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAV4</td>
        <td>Combined predominant swell wave period and height in feet ("4 group")</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAV5</td>
        <td>Combined secondary swell wave period and height in feet ("5 group")</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WPHM</td>
        <td>Combined wave period and height in half meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WVSW</td>
        <td>Combined swell wave direction, period and height in half meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SWEL</td>
        <td>Character combined swell wave direction, period and height in half meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DAWV</td>
        <td>Swell wave direction arrows of uniform length</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IDTH</td>
        <td>Thickness of ice on ship in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ROIA</td>
        <td>Rate of ice accretion on ship from WMO Code 3551</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IGRO</td>
        <td>Rate of ice accretion on vessel in salt water in inches per three hours</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DIGR</td>
        <td>Character rate of ice accretion in inches per three hours</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHPD</td>
        <td>True direction from which ship is moving (for 3 hours before obs) in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHPK</td>
        <td>Ship's average speed (for 3 hours before obs) in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DASH</td>
        <td>Ship's true direction arrows of uniform length</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PMN1</td>
        <td>Lowest 1-minute average pressure in previous hour in mb</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PMNT</td>
        <td>Time of lowest 1-minute average pressure, as hhmm</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PKWD</td>
        <td>Direction of 1-minute peak wind in previous hour in degrees</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PKWK</td>
        <td>Highest 1-minute mean wind speed in previous hour in knots</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PKWS</td>
        <td>Highest 1-minute mean wind speed in previous hour in m/sec</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PKWT</td>
        <td>Time of highest peak 1-minute wind in previous hour, as hhmm</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BRPK</td>
        <td>Wind barb (knots) for highest peak 1-minute wind</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Aircraft Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TURB</td>
        <td>Amount of turbulence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TBSE</td>
        <td>Base of turbulence in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TTOP</td>
        <td>Top of turbulence in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HBOT</td>
        <td>Base of turbulence in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HTOT</td>
        <td>Top of turbulence in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FQOT</td>
        <td>Frequency of turbulence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TPOT</td>
        <td>Type of turbulence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TBSY</td>
        <td>Graphics symbol for turbulence</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ICNG</td>
        <td>Amount of airframe icing</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IBSE</td>
        <td>Base of icing in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ITOP</td>
        <td>Top of icing in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HBOI</td>
        <td>Base of icing in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HTOI</td>
        <td>Top of icing in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TPOI</td>
        <td>Type of icing</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ICSY</td>
        <td>Graphics symbol for icing</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WBSE</td>
        <td>Base of weather in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WTOP</td>
        <td>Top of weather in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HBWX</td>
        <td>Base of weather in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HTWX</td>
        <td>Top of weather in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLC1</td>
        <td>Numeric cloud coverage 1</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CBS1</td>
        <td>Cloud base 1 in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTP1</td>
        <td>Cloud top 1 in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CB1M</td>
        <td>Cloud base 1 in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CT1M</td>
        <td>Cloud top 1 in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLC2</td>
        <td>Numeric cloud coverage 2</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CBS2</td>
        <td>Cloud base 2 in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTP2</td>
        <td>Cloud top 2 in feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CB2M</td>
        <td>Cloud base 2 in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CT2M</td>
        <td>Cloud top 2 in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ACRT</td>
        <td>Aircraft report type</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SELV</td>
        <td>Flight level in meters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FELV</td>
        <td>Flight level in hundreds of feet</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ITSY</td>
        <td>Icing type symbol</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TTSY</td>
        <td>Turbulence type symbol</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TFSY</td>
        <td>Turbulence frequency symbol</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ACTP</td>
        <td>Character aircraft type</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ATP1</td>
        <td>Numeric aircraft type</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Miscellaneous Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSBY</td>
        <td>Visibility in statute miles</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSBK</td>
        <td>Visibility in kilometers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSBN</td>
        <td>Visibility in nautical miles</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSBF</td>
        <td>Character visibility in fractions of statute miles for visibilities between 0. and 1.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VSBC</td>
        <td>Character visibility in fractions of statute miles for all visibility numbers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PnnI</td>
        <td>Precipitation over last nn hours in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PnnM</td>
        <td>Precipitation over last nn hours in millimeters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DPRC</td>
        <td>Character daily weather map precipitation in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PR24</td>
        <td>Precipitation over last 24 hours in inches, as sum of four successive 6-hour precip amounts</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNOW</td>
        <td>Snow depth in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNEW</td>
        <td>Amount of new snow in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNRT</td>
        <td>Forecast snow and ice pellet accumulation to watch threshold ratio</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SI12</td>
        <td>Forecast snow and ice pellet 12-h accumulation in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNIP</td>
        <td>Snow and ice pellet watch threshold in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FZRT</td>
        <td>Forecast freezing rain accumulation to watch threshold ratio</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FZ12</td>
        <td>Forecast Freezing rain 12-h accumulation in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FZRN</td>
        <td>Freezing rain watch threshold in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WEQS</td>
        <td>Water equivalent of snow on the ground in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HAIL</td>
        <td>Hail flag</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HLSZ</td>
        <td>Hail size in centimeters</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">DDEN</td>
        <td class="tg-implemented">Density of dry air in kg/(m**3)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.density.html#metpy.calc.density">metpy.calc.density(mixing=0)</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">PSYM</td>
        <td class="tg-implemented">Montgomery stream function in m**2/(100*s**2)</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.montgomery_streamfunction.html#metpy.calc.montgomery_streamfunction">metpy.calc.montgomery_streamfunction</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">HEAT</td>
        <td class="tg-implemented">Heat index in Fahrenheit</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.heat_index.html#metpy.calc.heat_index">metpy.calc.heat_index</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HMTR</td>
        <td>Humiture (apparent temperature) in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td class="tg-implemented">WCEQ</td>
        <td class="tg-implemented">Wind chill equivalent temperature in Fahrenheit</td>
        <td class="tg-implemented"><a href="api/generated/metpy.calc.windchill.html#metpy.calc.windchill">metpy.calc.windchill</a></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCHT</td>
        <td>Revised wind chill temperature in Fahrenheit</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MSUN</td>
        <td>Duration of sunshine in minutes</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FFnn</td>
        <td>Flash flood guidance for next nn hours in inches</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TOST</td>
        <td>Type of station (manned or automatic)</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STIM</td>
        <td>Report hour and minutes as hhmm</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TEXT</td>
        <td>Undecoded data</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SPCL</td>
        <td>Undecoded special reports</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MARK</td>
        <td>Markers</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FOSB</td>
        <td>Fosberg Index, also called Fire Weather Index</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Spacing Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BLNK</td>
        <td>Plot a blank, not accounted for in FILTER</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SPAC</td>
        <td>Plot a space, accounted for in FILTER</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td><b>Additional GEMPAK Parameters</b></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ADDSTN</td>
        <td>Logical variable which indicates whether stations which are in STNFIL, but not already included in the data file, should be added to the data file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AFOSFL</td>
        <td>The name of the AFOS graphics file to be displayed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AIRM</td>
        <td>The ending valid time for the airmet, the colors for the instrument flight rules, mountain obscuration, turbulence, icing, sustained winds and low-level wind shear, and flags for plotting symbols or airmet type, the end time, and the flight levels on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ANLYSS</td>
        <td>The average station spacing and the grid extend region.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ANOTLN</td>
        <td>The line attributes for annotation.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ANOTYP</td>
        <td>Specifies the fill type for annotation.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AREA</td>
        <td>The data area.  Only data within the area specified will be processed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ASCT</td>
        <td>The ending valid time for the AScat wind data, the speed intervals and colors, the wind barb size and width and plotting flags. Skip is a value that indicates how many rows and data points to skip when plotting.  The flags include High Wind Speed, Low Wind Speed, KNMI Quality Control Fail, Redundant Data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ASDI</td>
        <td>The ending valid time for ASDI data, the mode of the display, the time increments (in minutes going back from the ending time) and the corresponding colors for the ASDI data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ATCF</td>
        <td>The initial time for the ATCF forecast tracks, the colors for each model track, the model names, flags for plotting the time, the storm name or number, the forecast wind speeds and markers at each forecast time, and an optional specific storm identifier.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>AWPSFL</td>
        <td>The name of the AWIPS graphics file to be displayed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BND</td>
        <td>Specifies the parameters needed for processing bounds areas.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BORDER</td>
        <td>The color, line type and line width of the background and underground plot.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BOUNDS</td>
        <td>Specifies the bound area(s) to consider when performing the graph-to-grid procedure.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BOXLIN</td>
        <td>Line attributes for a box drawn around a region</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>BUFRFIL</td>
        <td>The BUFR output file names.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CALIMG</td>
        <td>Allows the user to select whether to use the calibration values supplied with an image, or raw pixel values.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CATMAP</td>
        <td>A string that contains "label=value" pairs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CBTOP</td>
        <td>User estimated cloud top height in meters. The default value is 8000 m.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CENTER</td>
        <td>Allows the GD2NDFD user to specify the originating or generating center ID and sub-center ID, as well as the Generating Process/Model ID.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CINT</td>
        <td>The contour interval, minimum and maximum values and number of digits.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLEAR</td>
        <td>A logical variable which determines whether the graphics screen is cleared before plotting.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CLRBAR</td>
        <td>Specifies the characteristics of a color bar associated with contour fill.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CNTRFL</td>
        <td>The name of the file containing contour information.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CNTR</td>
        <td>Specifies the plot attributes for Cell Centroid movement barbs (knots).</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CNTRPRM</td>
        <td>The SFPARM to contour.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>COLORS</td>
        <td>Specifies a list of color numbers.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>COLUMN</td>
        <td>Specifies the number of columns for plotting the contents of an ASCII text file specified by TXTFIL.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>COMPRESS</td>
        <td>A flag to determine whether the output will be written in compressed format.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CONTUR</td>
        <td>Sets attributes for the contour algorithms.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CPYFIL</td>
        <td>Identifies the location of the grid navigation and analysis information to be stored in a grid file, as well as an optional subarea.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CSIG</td>
        <td>The ending valid time for the convective sigmet and convective outlook, the colors for the initial hour (0-hr) CSIG, extrapolated 1-hr CSIG, extrapolated 2-hr CSIG, and outlook, and flags for plotting the 0-hr CSIG sequence number, end time, direction/speed, flight level, intensity, 1-hr CSIG sequence number, and 2-hr CSIG sequence number on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTLFLG</td>
        <td>A logical flag which indicates whether control characters are included in a raw surface data set to be decoded.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CTYPE</td>
        <td>Specifies the contouring algorithms to use.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CURVE</td>
        <td>A number corresponding to the method to be used to fit the curve.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CXSTNS</td>
        <td>Defines the x-axis for a cross-section plot.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>CYCLE</td>
        <td>The cycle name tag put into the VGF files.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DATOUT</td>
        <td>The date and time which will be assigned in the output file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DATTIM</td>
        <td>The date and time to be used by GEMPAK programs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DELTAN</td>
        <td>The average station spacing in degrees of latitude.  The Barnes objective analysis programs use this number to compute weights for data interpolation.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DELTAX</td>
        <td>The spacing between grid points in the x direction on CED grids. This value is in degrees longitude.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DELTAY</td>
        <td>The spacing between grid points in the y direction on CED grids. This value is in degrees latitude.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DELZ</td>
        <td>The user chosen average height difference between pressure levels to be used in the vertical interpolation. The default value is 500 m.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DESIRE</td>
        <td>The value of the output vertical coordinate variable to which the input variables should be interpolated.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DEVICE</td>
        <td>Specifies the graphics device.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DISCRETE</td>
        <td>A string that contains "value1-value2=value3" pairs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DITHER</td>
        <td>Used to specify the plotting behavior of the reflectivity intensities.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DLINES</td>
        <td>Indicates whether the direction of lines (order of points) should be taken into consideration when performing a graph-to-grid contour analysis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>DTAAREA</td>
        <td>Defines the area over which station data will be input to the Barnes objective analysis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ECHO</td>
        <td>Specifies whether to plot the grid box intesities.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EDGEOPTS</td>
        <td>An option which allows users to specify boundary conditions for the analysis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EDR</td>
        <td>Starts with the ending valid time for EDR data followed by corresponding colors for the EDR data over the time limit specified.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ENCY</td>
        <td>The initial time for the ENCY forecast tracks, the colors for each model track, the model names, flags for plotting the time, the forecast pressures and markers at each forecast time.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>EXTEND</td>
        <td>Specifies the numbers of grid points beyond the GRDAREA which define the grid extend area in the Barnes objective analysis.  The first pass is computed on the extend area to reduce edge effects on the GRDAREA.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FAXFIL</td>
        <td>The name of a 6-bit FAX product.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FFA</td>
        <td>The ending valid time for the flood watches, the colors for the flash flood and areal flood watches, a flag for plotting the start and stop times, a flag for plotting the zone names for the storms on the map, a flag for plotting the immediate cause for the flooding, and a flag to outline the zone.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FHOUR</td>
        <td>The forecast hour, e.g., 18, or 24, which defines the "f" value in the BUFRFIL name.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FHR</td>
        <td>The forecast hour of freezing levels. FHR can be a single hour or a range.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FILTER</td>
        <td>A logical variable or real number which controls the filtering of data in order to eliminate plotting of overlapping data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FILTYP</td>
        <td>The filter type.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FINT</td>
        <td>The contour fill interval, minimum and maximum values.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FLINE</td>
        <td>The color and fill type to be used for contour fill.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>FXYTBL</td>
        <td>The FXY table file names.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2DIAG</td>
        <td>Allows for detailed GRIB2 message section information, entry-by-entry, to be printed out for selected GRIB2 messages.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2DRT</td>
        <td>Specifies the scaling and packing options when encoding the GRIB2 message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2IDS</td>
        <td>A list of integers that are encoded into the GRIB2 Identification Section ( Section 1 ) that identifies the source and type of data packed in the GRIB2 message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2IS</td>
        <td>A list of integers that are encoded into the GRIB2 Indicator Section ( Section 0 ) that identify the discipline and version of the GRIB2 message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2PDT</td>
        <td>Can be used to specify any or all of the values in the Product Definition Template (PDT), describing the grid in the output GRIB2 message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>G2TBLS</td>
        <td>Allows for specification of the GRIB2 decoding tables.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GAIRM</td>
        <td>Makes provision to plot the G-AIRMET snapshots in 3-hour time bins.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GAMMA</td>
        <td>The convergence parameter, is a multiplier for the weight and search radius for passes after the first pass of the Barnes analysis programs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GAREA</td>
        <td>The graphics area.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GBDIAG</td>
        <td>Allows for detailed GRIB message section information, byte-by-byte, to be printed out for selected GRIB messages.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GBFILE</td>
        <td>The name of the file which contains gridded data in GRIB messages.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GBTBLS</td>
        <td>Allows for specification of the GRIB decoding tables.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GCENTER</td>
        <td>Sets the center latitude-longitude on the ETA model domain.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDATTIM</td>
        <td>The date/time for the grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDEFIL</td>
        <td>The name of the grid edit file which will be used to update a grid file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDFILE</td>
        <td>The name of the file which contains gridded data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDNUM</td>
        <td>Allows the user to select grids by number.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDOUTF</td>
        <td>The output grid data file name.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDOUTL</td>
        <td>The output vertical level in the target vertical coordinate.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GDPFUN</td>
        <td>Specifies a grid diagnostic function which yields either a scalar or vector quantity.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GFUNC</td>
        <td>Specifies a grid diagnostic function which yields a scalar quantity.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GGLIMS</td>
        <td>The parameter which controls the grid value limits and values.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GGVGF</td>
        <td>The name of the VGF file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GLEVEL</td>
        <td>The vertical level for the grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GPACK</td>
        <td>The packing type and the number of bits (or data precision) to be used to pack the grid data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GPOINT</td>
        <td>The grid location to be used for the plot.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GRDAREA</td>
        <td>Specifies the area to be covered by the grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GRDHDR</td>
        <td>A list of the valid grid header flags.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GRDLBL</td>
        <td>The color number to be used in plotting the grid index numbers.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GRDNAM</td>
        <td>The parameter name for the grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GRDTYP</td>
        <td>The type of a diagnostic grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GSKIP</td>
        <td>Allows the NDFDG2 user to specify a skip factor in order to reduce the resolution of the resulting GEMPAK grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GSPACE</td>
        <td>Sets the grid spacing of the ETA model domain.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GUESFUN</td>
        <td>Specifies one or more grid diagnostic functions that compute the first guess grid for a Barnes objective analysis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GUESS</td>
        <td>The name of the file which contains the first guess gridded data for objective analysis programs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GVCORD</td>
        <td>The vertical coordinate of the grid to be selected.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GVECT</td>
        <td>Specifies a grid diagnostic function which yields a vector quantity.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>GVOUTC</td>
        <td>The vertical coordinate of the output grid.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HILO</td>
        <td>Contains the information for plotting relative highs and lows.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HISTGRD</td>
        <td>Toggles the writing of the graph-to-grid history grid to the GEMPAK grid file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HLSYM</td>
        <td>Defines the characteristics for the HILO symbols.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>HRCN</td>
        <td>The ending valid time for the tropical disturbance, the colors for the hurricanes, tropical storms, tropical depressions and directional arrows, the symbols for the hurricanes, tropical storms and tropical depressions, and flags for plotting the center located time, the name and minimum central pressure, the speed, the wind and sea quadrant radii, and the forecast track on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IDNTYP</td>
        <td>Sets character or numeric station identifiers to be used for input or output.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IJSKIP</td>
        <td>Used to control subsetting of the internal grid by declaring bounding index values and numbers of points to skip in each index direction, I and J.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IMCBAR</td>
        <td>Specifies the characteristics of a color bar for images.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IMGFIL</td>
        <td>The name of an image file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IMGTIM</td>
        <td>The date and time to be used.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>IMJM</td>
        <td>Sets the number of grid points for workstation eta in the N-S and E-W direction.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>INDXFL</td>
        <td>The name of the file which contains the GRIB message header information.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>INFO</td>
        <td>The information needed to define the object to be plotted.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>INTERP</td>
        <td>A logical variable which determines whether interpolation between sweeps will occur.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ISIG</td>
        <td>The ending valid time for the international SIGMET, the colors for the thunderstorms, turbulence, hurricanes, tropical storms, tropical  depressions, volcanic ash clouds, marked mountain waves, tropical cyclones, squall lines, CAT, icing, hail, duststorms, sandstorms, cumulonimbus, and low level wind shear, flags for plotting symbols or storm names, the start and end times, the message id, the direction and speed, and the flight level or the central pressure and maximum wind speed associated with a tropical cyclone on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>KEYCOL</td>
        <td>Indicates which contour lines to process.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>KEYLINE</td>
        <td>Indicates which contour lines to process based on VGTYPE and SUBTYP.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>KXKY</td>
        <td>Specifies the size of a grid as two numbers.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LAT</td>
        <td>Specifies the latitude grid lines to be drawn.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LATLON</td>
        <td>Specifies the latitude and longitude grid lines to be drawn.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LEVELS</td>
        <td>Specifies the vertical levels to be extracted from the data set.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LINE</td>
        <td>The color, line type, line width, line label frequency, smoothing separated by slashes, and flag to suppress small contours.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LOCI</td>
        <td>The point(s) needed to place the object to be plotted.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LON</td>
        <td>Specifies the longitude grid lines to be drawn.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LSTALL</td>
        <td>A flag indicating whether the full contents of a file are to be listed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LSTPRM</td>
        <td>Specifies a field to list on the side of the display. Stations are added to the list if they are filtered from the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LTNG</td>
        <td>The ending valid time for lightning data, the time increments (in minutes going back from the ending time) and the corresponding colors for the lightning data, and the positive and negative markers to display.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LUTFIL</td>
        <td>Specifies a lookup table file name used to enhance the colors for satellite or radar images.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>LVLINCR</td>
        <td>Specifies the freezing level increment.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MAPFIL</td>
        <td>The name(s) of the map file(s) to be used for maps drawn by GEMPAK programs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MAP</td>
        <td>The map color, line type and line width.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MARKER</td>
        <td>Specifies the marker color, type, size, line width and hardware/software flag.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MAXGRD</td>
        <td>The maximum number of grids that can be stored in the grid file being created.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MAXTOP</td>
        <td>Specifies the color and filter attributes for cell top annotations (feet).</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MESO</td>
        <td>Specifies the plot symbol and filter attributes for mesocyclones.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MIXRLN</td>
        <td>Specifies the color, line type, line width, minimum, maximum, and increment for the background mixing ratio lines on thermodynamic diagrams.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MRGDAT</td>
        <td>A logical variable indicating whether sounding data is to be merged or unmerged.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MSCALE</td>
        <td>Specifies the characteristics of a scale legend associated with map projections.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>MXDPTH</td>
        <td>The user estimated mixed layer depth in meters.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NCON</td>
        <td>The ending valid time for the non-convective sigmet, the colors for the icing, turbulence, duststorm and sandstorm, and volcanic ash, and flags for plotting symbols, the end time, the message id, and the flight levels on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NDVAL</td>
        <td>The data value to be used where NIDS products report "ND" (none detected).</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NPASS</td>
        <td>Controls the number of passes for the Barnes objective analysis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>NTRACE</td>
        <td>The number of traces to be drawn in SFGRAM.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OAATTR</td>
        <td>Contains attributes to use for objective analysis in the graph to grid function.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OABND</td>
        <td>Specifies the bounds file(s) to use for 'blocking'.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OAGUESS</td>
        <td>Contains the information to use as a first guess for objective analysis in the graph to grid function.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OLKDAY</td>
        <td>The day of the extended oulook.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OSCT</td>
        <td>The ending valid time for the OScat wind data, the speed intervals and colors, the wind barb size and width and plotting flags.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OUTFIL</td>
        <td>The name of an output satellite image file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OUTPUT</td>
        <td>Determines the output devices.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>OVERWR</td>
        <td>Allows the user to specify whether existing GEMPAK grids in a file should be overwritten or left unchanged.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PANEL</td>
        <td>Specifies the panel location, panel outline color, line type and width.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PDSEXT</td>
        <td>A logical flag which only becomes applicable when a PDS extension exists in the GRIB message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PDSVAL</td>
        <td>Provides a way to enter explicitly PDS numbers identifying a grid by parameter, level, vertical coordinate type, and time.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PIXRES</td>
        <td>The input for how many pixels and lines to include in the new image.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PLUS</td>
        <td>Specifies the size and width of a plus sign.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PMAX</td>
        <td>Defines the maximum possible pressure from which data may be interpolated to the output vertical coordinate.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>POSN</td>
        <td>The position number and the format of the text to be used to plot data in GDMAP.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PRBTYP</td>
        <td>Specifies the probability type.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PRECSN</td>
        <td>The binary or decimal precision to be preserved in the packing of the gridded data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PROJ</td>
        <td>The map projection, projection angles, and margins.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>PTYPE</td>
        <td>The type of y axis plot to be used, the height-to-width ratio of the plot, and the margins.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QCNTL</td>
        <td>The quality control threshold values.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>QSCT</td>
        <td>The ending valid time for the QuikScat or ambiguity wind data, the speed intervals and colors, the wind barb size and width and plotting flags.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADDUR</td>
        <td>The length of time (in minutes) prior to RADTIM for which data will be used in composites.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADFIL</td>
        <td>The name of a radar image file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADFRQ</td>
        <td>tThe update frequency for RADAR composites.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADINFO</td>
        <td>Specifies the color for radar site operational status annotations.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADIUS</td>
        <td>The search radius (in meters) for which data will be considered.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADMODE</td>
        <td>Allows the user to select whether to include radar data from sites operating in (P) precipitation/storm mode, (C) clear air mode, and/or (M) maintainence mode.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADPARM</td>
        <td>The Radar parameter to be displayed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RADTIM</td>
        <td>The date and time to be used for RADAR composites.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RCMFIL</td>
        <td>The RCM data file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>REFVEC</td>
        <td>Specifies the size and location on the screen of the.reference arrow.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>REGION</td>
        <td>Specifies an areal location.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RESPOND</td>
        <td>A logical variable indicating whether the user will respond interactively to GEMPAK programs.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>ROTATE</td>
        <td>The angle of rotation for the coordinate axes.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>RTRAJ</td>
        <td>A logical variable which determines whether the trajectory will start or end at the specified GPOINT.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SATFIL</td>
        <td>The name of a satellite image file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SCALE</td>
        <td>The scaling factor for the data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SEARCH</td>
        <td>Controls the search radius in an objective analysis program.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFEFIL</td>
        <td>The name of the surface edit file to be used to update a surface file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFFILE</td>
        <td>The name of the surface data file to be accessed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFFSRC</td>
        <td>The surface file source.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFOUTF</td>
        <td>The output surface data file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFPARM</td>
        <td>A list of surface parameters to be used in a surface program.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SFPRMF</td>
        <td>Specifies the packing information for the surface file to be created.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SGWH</td>
        <td>The ending valid time for the significant wave height data, the height intervals and colors.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHAPE</td>
        <td>The object that the user wishes to plot on the current graphics device.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SHIPFL</td>
        <td>A logical variable which indicates whether the surface file contains stations which are not at a fixed location, such as moving ships, aircraft, or floating buoys.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKIP</td>
        <td>A variable which determines the contour points or plot points to skip.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SKPMIS</td>
        <td>A logical variable which indicates whether non-reporting stations will be listed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNBUFR</td>
        <td>The name of the BUFR model sounding file to be used as input to create GEMPAK sounding and surface data files using program NAMSND.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNEFIL</td>
        <td>The name of the sounding edit file to be used to update a sounding data file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNFILE</td>
        <td>The filename for an upper air data set.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNOUTF</td>
        <td>The output sounding data file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNPARM</td>
        <td>A list of upper air parameters to be used in an upper-air program.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SNPRMF</td>
        <td>Specifies the packing formation for the sounding file to be created.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SOURCE</td>
        <td>Indicates whether the data used to compute the average station spacing are to be read from a surface or sounding file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SPLINE</td>
        <td>A logical for using splines to interpolate the data to height levels.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SQUALL</td>
        <td>The length of the squall line used for the air and moisture flux calculations.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STARTL</td>
        <td>The level in the input data file at which to begin the search for the output vertical level in gdpvsf.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STAT</td>
        <td>The issuing status of the GFA freezing levels.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STATION</td>
        <td>The station to use in SFGRAM.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNCOL</td>
        <td>Specifies the color for the station identifier, time and the parameters specified in STNDEX.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNDEX</td>
        <td>The list of stability indices or station parameters for upper-air data.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNFIL</td>
        <td>The name of a file which contains station information which includes the character identifier, number, name, state, country, latitude, longitude and elevation for each station.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNPLT</td>
        <td>Allows the user to plot station markers and station information.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STNTYP</td>
        <td>Used to select the data reporting characteristic of a station.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STOPL</td>
        <td>The level in the input data file at which to end the search for the output vertical level in gdpvsf.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STREAM</td>
        <td>Controls several parameters dealing with the overall streamline calculation and display.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>STRMID</td>
        <td>Specifies the storm identifier used by programs GPTPC and GPTCWW.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SVRL</td>
        <td>The ending valid time for the SLS watches, the colors for the severe thunderstorm and tornado (SLS) watches, a flag for plotting the start and stop times, a flag for plotting the county names for the watches on the map, and a flag to outline the county.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>SYSTEM</td>
        <td>The system (storm) speed (m/s) and direction.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TAG</td>
        <td>Used to identify a group of GFA FZLVL elements.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TAXIS</td>
        <td>Contains the range, increment and location for labels on a time axis.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TCMG</td>
        <td>The ending valid time for the tropical disturbance, the colors for the disturbance symbol, the arrows, and the storm danger area, and the name of the center issuing the graphic.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TEXT</td>
        <td>The size, font, text width and hardware/software flag for graphics text.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTALN</td>
        <td>Specifies the color, line type, line width, minimum, maximum, and increment for the background dry adiabats (potential temperature lines) on thermodynamic diagrams</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>THTELN</td>
        <td>Specifies the color, line type, line width, minimum, maximum, and increment for the background moist adiabats (equivalent potential temperature lines) on thermodynamic diagrams</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TILT</td>
        <td>The Radar beam elevation/tilt angle.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TIMSTN</td>
        <td>Contains the maximum number of times to include in a file and the number of stations to be included in addition to the stations in STNFIL.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TITLE</td>
        <td>The title color, title line, and title string.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TOPOFL</td>
        <td>The topographic input file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TRACE</td>
        <td>Specifications for each trace on the meteogram.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TRAK</td>
        <td>The ending time for the Altimetric Satellite Ground Track Prediction data, and the color of the prediction.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TROPHT</td>
        <td>The user estimated tropopause height in meters.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TRPINT</td>
        <td>The user chosen distance above (and below) the tropopause which is used for layer calculations.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TSTEP</td>
        <td>Specifies the time step, in minutes, for the calculation of updated position of the parcel within the grid domain.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TVS</td>
        <td>Specifies the plot symbol and filter attributes for tornado vortex signatures.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TXTCOL</td>
        <td>Specifies the color number for text, NOAA or NWS logo.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TXTFIL</td>
        <td>Specifies an ASCII text file to be read and displayed to the current device driver.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TXTLOC</td>
        <td>Specifies the start location for plotting the contents of an ASCII text file specified by TXTFIL.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TXTYPE</td>
        <td>Specifies the text attributes.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>TYPE</td>
        <td>Specifies the processing type for the GDPLOT2 GDPFUN parameter.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>UKAFIL</td>
        <td>The intermediate input/output ASCII file.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VCOORD</td>
        <td>Specifies the vertical coordinate system of the levels to process.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VERCEN</td>
        <td>Allows the GDGRIB user to specify the contents of bytes 4, 5, 6, and 26 of the GRIB PDS.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>VGFILE</td>
        <td>The name of the Vector Graphics File (VGF) to be displayed or processed.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WARN</td>
        <td>The ending valid time for the warnings, the colors for the severe thunderstorm, tornado and flash flood warnings, a flag for plotting the start and stop times, a flag for plotting the county names for the warning on the map, a flag to outline the county, and a flag to plot warning polygon.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WATCH</td>
        <td>The ending valid time for the watches, the colors for the severe thunderstorm and tornado watches and a flag for plotting the start and stop times for the watch on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAVLEN</td>
        <td>the wavelength for the gravity or lee wave in kilometers.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WAVSPD</td>
        <td>The wave speed in m/s. This is used for calculating the SCORER parameter.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCN</td>
        <td>The ending valid time for the watch county notification(WCN), the colors for the county bounds, a flag for plotting the start and stop times, a flag for plotting the county names for the WCN on the map, a flag to outline the county or union, a flag to fill the county or union, and a union flag.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WCP</td>
        <td>The ending valid time for the watches, the colors for the severe thunderstorm and tornado watches and flags for plotting the start and stop times and watch numbers for the watches on the map.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WEIGHT</td>
        <td>The Barnes weighting parameter.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WIND</td>
        <td>Specifies the wind symbol, size, width, type, and head size.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WINPOS</td>
        <td>Specifies the position for plotting winds for vertical profile plots.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WMOHDR</td>
        <td>Allows specification of a WMO header for a GRIB message.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WOU</td>
        <td>The ending valid time for the watch outline update (WOU), the colors for the county bounds, a flag for plotting the start and stop times, a flag for plotting the county names for the WOU on the map, a flag to outline the county or union, a flag to fill the county or union, and a union flag.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSAT</td>
        <td>The ending valid time for the WindSAT wind data, the speed intervals and colors, the wind barb size and width and plotting flags.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSPDA</td>
        <td>The ending valid time for the Altika wind speed data, the speed intervals and colors, and time stamp intervals and colors.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSPDALT</td>
        <td>The ending valid time for the altimeter-derived wind speed data, the speed intervals and colors, and time stamp intervals and colors.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>WSTM</td>
        <td>The ending valid time for the winter storms, the colors for the storm warning, watch and advisory, a flag for plotting the start and stop times, a flag for plotting the zone names for the storms on the map, and a flag to outline the zone.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>XAXIS</td>
        <td>Contains the left bound, right bound, label increment, and frequency information.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
      <tr>
        <td>YAXIS</td>
        <td>Contains the lower bound, upper bound, label increment, and frequency information.</td>
        <td></td>
        <td></td>
        <td></td>
        <td></td>
      </tr>
    </table>

