# Contributors Guide

Interested in helping build MetPy? Have code from your research that you believe others will
find useful? Have a few minutes to tackle an issue? In this guide we will get you setup and
integrated into contributing to MetPy!

## Introduction
First off, thank you for considering contributing to MetPy. MetPy is community-driven
project. It's people like you that make MetPy useful and successful. There are many ways
to contribute, from writing tutorials or examples, improvements to the documentation,
submitting bug reports and feature requests, or even writing code which can be incorporated
into MetPy for everyone to use.

Following these guidelines helps to communicate that you respect the time of the
developers managing and developing this open source project. In return, they
should reciprocate that respect in addressing your issue, assessing changes, and
helping you finalize your pull requests.

So, please take a few minutes to read through this guide and get setup for success with your
MetPy contributions. We're glad you're here!

## What Can I Do?
* Tackle any [issues](https://github.com/Unidata/MetPy/issues) you wish! We have a special
  label for issues that beginners might want to try. Have a look at our
  [current beginner issues.](https://github.com/unidata/metpy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Difficulty%3A+Beginner%22)
  Also have a look at if the issue is already assigned to someone - this helps us make sure
  that work is not duplicated if the issue is already being worked on by Unidata Staff.

* Contribute code you already have. It does not need to be perfect! We will help you clean
  things up, test it, etc.

* Make a tutorial or example of how to do something.

* Improve documentation of a feature you found troublesome.

* File a new issue if you run into problems!

## Ground Rules
The goal is to maintain a diverse community that's pleasant for everyone. Please
be considerate and respectful of others by following our 
[code of conduct](https://github.com/Unidata/MetPy/blob/master/CODE_OF_CONDUCT.md). 

Other items:

* Each pull request should consist of a logical collection of changes. You can
  include multiple bug fixes in a single pull request, but they should be related.
  For unrelated changes, please submit multiple pull requests.
* Do not commit changes to files that are irrelevant to your feature or bugfix
  (eg: .gitignore).
* Be willing to accept criticism and work on improving your code; we don't want
  to break other users' code, so care must be taken not to introduce bugs.
* Be aware that the pull request review process is not immediate, and is
  generally proportional to the size of the pull request.

## Reporting a bug
The easiest way to get involved is to report issues you encounter when using MetPy or by
requesting something you think is missing.

* Head over to the [issues](https://github.com/Unidata/MetPy/issues) page.
* Search to see if your issue already exists or has even been solved previously.
* If you indeed have a new issue or request, click the "New Issue" button.
* Fill in as much of the issue template as is relevant. Please be as specific as possible. 
  Include the version of the code you were using, as well as what operating system you 
  are running. If possible, include complete, minimal example code that reproduces the problem.

## Setting up your development environment
We recommend using the [conda](https://conda.io/docs/) package manager for your Python 
environments. Our recommended setup for contributing is:

* Install [miniconda](https://conda.io/miniconda.html) on your system.
* Install git on your system if it is not already there (install XCode command line tools on
  a Mac or git bash on Windows)
* Login to your GitHub account and make a fork of the 
  [MetPy repository](https://github.com/unidata/metpy/) by clicking the "Fork" button.
* Clone your fork of the MetPy repository (in terminal on Mac/Linux or git shell/
  GUI on Windows) in the location you'd like to keep it. We are partial to creating a
  ``git_repos`` directory in our home folder.
  ``git clone https://github.com/your-user-name/metpy.git``
* Navigate to that folder in the terminal or in Anaconda Prompt if you're on Windows.
  ``cd metpy``
* Connect your repository to the upstream (main project).
  ``git remote add unidata https://github.com/unidata/metpy.git``
* Create the development environment by running ``conda env create``. This will install
  all of the packages in the ``environment.yml`` file.
* Activate our new development environment ``source activate devel`` on Mac/Linux or
  ``activate devel`` on Windows.
* Make an editable install of MetPy by running ``pip install -e .``

Now you're all set! You have an environment called ``devel`` that you can work in. You'll need
to make sure to activate that environment next time you want to use it after closing the
terminal or your system. If you want to get back to the root environment, just run
``source deactivate`` (just ``deactivate`` on Windows).

## Pull Requests

The changes to the MetPy source (and documentation) should be made via GitHub pull requests
against ``master``, even for those with administration rights. While it's tempting to
make changes directly to ``master`` and push them up, it is better to make a pull request so
that others can give feedback. If nothing else, this gives a chance for the automated tests to
run on the PR. This can eliminate "brown paper bag" moments with buggy commits on the master
branch.

During the Pull Request process, before the final merge, it's a good idea to rebase the branch
and squash together smaller commits. It's not necessary to flatten the entire branch, but it
can be nice to eliminate small fixes and get the merge down to logically arranged commits. This
can also be used to hide sins from history--this is the only chance, since once it hits
``master``, it's there forever!

**Working on your first Pull Request?** You can learn how from this *free* video series
[How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github), Aaron Meurer's [tutorial on the git workflow](https://www.asmeurer.com/git-workflow/), or the guide [“How to Contribute to Open Source"](https://opensource.guide/how-to-contribute/).

Commit the changes you made. Chris Beams has written a [guide](https://chris.beams.io/posts/git-commit/) on how to write good commit messages.

Push to your fork and [submit a pull request]( https://github.com/Unidata/metpy/compare/).
For the Pull Request to be accepted, you need to agree to the
MetPy Contributor License Agreement (CLA). This will be handled automatically
upon submission of a Pull Request.
See [here](https://github.com/Unidata/MetPy/blob/master/CLA.md) for more
explanation and rationale behind MetPy's CLA.

## Documentation
Now that you've made your awesome contribution, it's time to tell the world how to use it.
Writing documentation strings is really important to make sure others use your functionality
properly. Didn't write new functions? That's fine, but be sure that the documentation for
the code you touched is still in great shape. It is not uncommon to find some strange wording
or clarification that you can take care of while you are here. If you added a new function
make sure that it gets marked as included if appropriate in the GEMPAK conversion table.

You can write examples in the documentation if they are simple concepts to demonstrate. If
your feature is more complex, consider adding to the examples or tutorials for MetPy.

You can build the documentation locally to see how your changes will look.
* Navigate to the docs folder ``cd docs``
* Remove any old builds and build the current docs ``make clean html``
* Open ``docs/build/html/index.html`` and see your changes!

## Tests
Unit tests are the lifeblood of the project, as it ensures that we can continue to add and
change the code and stay confident that things have not broken. Running the tests requires
``pytest``, which is easily available through ``conda`` or ``pip``. It was also installed if
you made our default ``devel`` environment. 

### Running Tests
Running the tests can be done by running ``py.test``

Running the whole test suite isn't that slow, but can be a burden if you're working on just
one module or a specific test. It is easy to run tests on a single directory:

    py.test metpy/calc
    
A specific test can be run as:

    py.test -k test_my_test_func_name

### Writing Tests
Tests should ideally hit all of the lines of code added or changed. We have automated
services that can help track down lines of code that are missed by tests. Watching the
coverage has even helped find sections of dead code that could be removed!

Let's say we are adding a simple function to add two numbers and return the result as a float
or as a string. (This would be a silly function, but go with us here for demonstration
purposes.)

    def add_as_float_or_string(a, b, as_string=False):
        res = a + b
        if as_string:
           return string(res)
        return res
        
I can see two easy tests here: one for the results as a float and one for the results as a
string. If I had added this to the ``calc`` module, I'd add those two tests in 
``tests/test_calc.py``.

    def test_add_as_float_or_string_defaults():
        res = add_as_float_or_string(3, 4)
        assert(res, 7)
        
    
    def test_add_as_float_or_string_string_return():
        res = add_as_float_or_string(3, 4, as_string=True)
        assert(res, '7')

There are plenty of more advanced testing concepts, like dealing with floating point
comparisons, parameterizing tests, testing that exceptions are raised, and more. Have a look
at the existing tests to get an idea of some of the common patterns.

### Image tests
Some tests (for matplotlib plotting code) are done as an image comparison, using the
pytest-mpl plugin. To run these tests, use:

    py.test --mpl

When adding new image comparison tests, start by creating the baseline images for the tests:

    py.test --mpl-generate-path=baseline

That command runs the tests and saves the images in the ``baseline`` directory. 
For MetPy this is generally ``metpy/plots/tests/baseline/``. We recommend using the ``-k`` flag
to run only the test you just created for this step.

For more information, see the `docs for mpl-test <https://github.com/astrofrog/pytest-mpl>`_.


## Cached Data Files
MetPy keeps some test data, as well as things like shape files for US counties in a data cache
supported by the pooch library. To add files to this, please ensure they are as small as
possible. Put the files in the `staticdata` directory. Then run this command in the metpy
directory (that contains the `static-data-manifest.txt` file)to recreate the data registry:

`python -c "import pooch; pooch.make_registry('staticdata', 'metpy/static-data-manifest.txt')"`

Make sure that no system files (like `.DS_Store`) are in the manifest and add it to your
contribution.

## Code Style
MetPy uses the Python code style outlined in `PEP8
<http://pep8.org>`_. For better or worse, this is what the majority
of the Python world uses. The one deviation is that line length limit is 95 characters. 80 is a
good target, but some times longer lines are needed.

While the authors are no fans of blind adherence to style and so-called project "clean-ups"
that go through and correct code style, MetPy has adopted this style from the outset.
Therefore, it makes sense to enforce this style as code is added to keep everything clean and
uniform. To this end, part of the automated testing for MetPy checks style. To check style
locally within the source directory you can use the ``flake8`` tool. Running it
from the root of the source directory is as easy as running ``pytest --flake8`` in the base
of the repository.
    
You can also just submit your PR and the kind robots will comment on all style violations as
well. It can be a pain to make sure you have the right number of spaces around things, imports
in order, and all of the other nits that the bots will find. It is very important though as
this consistent style helps us keep MetPy readable, maintainable, and uniform.

## What happens after the pull request
You've make your changes, documented them, added some tests, and submitted a pull request.
What now? 

### Automated Testing
First, our army of never sleeping robots will begin a series of automated checks.
The test suite, documentation, style, and more will be checked on various versions of Python
with current and legacy packages. Travis CI will run testing on Linux and Mac, Appveyor will
run tests on Windows. Other services will kick in and check if there is a drop in code coverage
or any style variations that should be corrected. If you see a red mark by a service, something
failed and clicking the "Details" link will give you more information. We're happy to help if
you are stuck.

The robots can be difficult to satisfy, but they are there to help everyone write better code.
In some cases, there will be exceptions to their suggestions, but these are rare. If you make
changes to your code and push again, the tests will automatically run again.

### Code Review
At this point you're waiting on us. You should expect to hear at least a comment within a
couple of days. We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write tests.
* Follow [PEP8][http://pep8.org] for style. (The `flake8` utility can help with this.)
* Write a [good commit message][https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html].

Pull requests will automatically have tests run by Travis. This includes
running both the unit tests as well as the `flake8` code linter.

### Merging
Once we're all happy with the pull request, it's time for it to get merged in. Only the
maintainers can merge pull requests and you should never merge a pull request you have commits
on as it circumvents the code review. If this is your first or second pull request, we'll
likely help by rebasing and cleaning up the commit history for you. As your developement skills
increase, we'll help you learn how to do this.


## More Questions?
If you're stuck somewhere or are interested in being a part of the community in
other ways, feel free to contact us:
* [MetPy's Gitter Channel](https://gitter.im/Unidata/MetPy)
* ["metpy" tag on Stack Overflow](https://stackoverflow.com/questions/tagged/metpy)
* [Unidata's Python support address](mailto:support-python@unidata.ucar.edu)
* [python-users](https://www.unidata.ucar.edu/support/#mailinglists) mailing list

## Futher Reading
There are a ton of great resources out there on contributing to open source and on the
importance of writing tested and maintainable software.
* [GitHub's Contributing to Open Source Guide](https://guides.github.com/activities/contributing-to-open-source/)
* [Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
