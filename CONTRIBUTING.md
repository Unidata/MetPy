# Contributing

## Issues
When creating a new issue, please be as specific as possible. Include the version of the code you were using.
If possible, include complete, minimal example code that reproduces the problem.

## Pull Requests
We love pull requests from everyone. Fork, then clone the repo:

    git clone git@github.com:your-username/metpy.git

Install metpy:

    python setup.py install

Install py.test and make sure the tests pass:

    pip install pytest
    py.test

Make your change. Add tests for your change. Make the tests pass:

    py.test

Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/metpy/metpy/compare/

At this point you're waiting on us. You should expect to hear at least a comment within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted:

* Write tests.
* Follow [PEP8][pep8] for style. (The `flake8` utility can help with this.)
* Write a [good commit message][commit].

Pull requests will automatically have tests run by Travis. This includes running both the unit
tests as well as the `flake8` code linter.

[pep8]: https://www.python.org/dev/peps/pep-0008/
[commit]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
