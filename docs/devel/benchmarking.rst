========================
Performance Benchmarking
========================

This guide provides information on the implementation and management of benchmarking in MetPy.

-----------------
Airspeed Velocity
-----------------

MetPy's source code is benchmarked using `Airspeed Velocity <https://github.com/airspeed-velocity/asv>`_.
ASV is an open source software which builds environments based on historical and current
iterations of software and runs benchmark functions before compiling the results into
digestable html pages. MetPy's developers have used GitHub Actions and a Unidata Jenkins
instance in order to automatically perform benchmarking as part of the continuous
integration/continuous development workflow. These benchmarks allow us to identify bottlenecks
in the code, determine performance changes from pull requests, and view a history of MetPy's
time efficiency.

----------------------
Historical Performance
----------------------

Results of Metpy's performance throughout versions is available at `this page <https://unidata.github.io/metpy-benchmark/#>`_.
Currently, the history is benchmarked starting with MetPy version ``1.4.0`` and benchmarks the
first commit hash associated with each minor version until present. Additionally, starting with
the most recent minor release, every merged pull request made by a human contributor is
benchmarked and added to the results. Note that these benchmarks run weekly, so it may take a
few days for your merged commit to be updated into the results.

This performance history is run using the Unidata Jenkins instance. Upon run, the
``benchmarks/Jenkinsfile`` instructs the Jenkins instance to create a custom
``Docker container`` using the ``benchmarks/Dockerfile`` and runs the benchmark
functions within it. Jenkins uses the same Unidata machine for each run in order to ensure
consistent benchmarking results. ASV is installed in this container and runs the benchmark
functions for the historical commits of interest. In the event that successful results already
exist for the requested commit hash, ASV will skip it and maintain the previous results.
Finally, Jenkins pushes the results to a separate `results repository <https://github.com/unidata/metpy-benchmark>`_
where a GitHub Action uses an ASV command to generate and deploy the html.

-------------------
Benchmark Functions
-------------------

Located within the ``benchmarks/benchmarks`` directory are ``.py`` files each containing a
class ``TimeSuite``, ``setup`` and ``setup_cache`` functions, and functions with the name
scheme ``time_example_metpy_function``. This is ASV's required `syntax <https://asv.readthedocs.io/en/latest/writing_benchmarks.html>`_
for writing benchmarks. The ``setup_cache`` function loads the artificial benchmarking dataset
``data_array_compressed.nc`` and prepares the dataset for use by the benchmarks. The ``setup``
function "slices" the 4D dataset into the appropriate dimensions to create variables that can
be passed to and used by the benchmark functions. Each benchmarking function then receives one
of these slices (or the entire dataset) and runs the code inside the function as a benchmark,
timing the performance and saving the results.

------------------
Local Benchmarking
------------------

If you would like to run the benchmarking suite on your own development branch,
follow these steps:

1. Install asv in your ``devel`` environment using ``conda install asv``
2. Ensure that you have the ``benchmarks`` directory at the root of your MetPy repository
3. Navigate to the ``benchmarks`` directory: ``cd benchmarks``
4. Now it depends on exactly which benchmarks you want to run:

    a. To benchmark your code as is currently is, use ``python -m asv run``
    b. To compare a working branch with *your version* of MetPy's main branch, use
    ``python -m av continuous main branch_name`` where ``branch name`` is the name of your
    branch. You can also simply use two commit hashes in the place of the branch names.
    c. To run the history of MetPy as mentioned above, you can use
    ``python -m asv run HASHFILE:no_bot_merge_commits.txt``.
    **Note that this is somewhat computationally taxing and often takes several hours, depending on the specs of your machine**