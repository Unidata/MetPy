# Contributing

## Introduction
First off, thank you for considering contributing to MetPy. MetPy is community-driven
project, so it's people like you that make MetPy useful and successful.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

We love contributions from community members, just like you! There are many ways
to contribute, from writing tutorial or example Jupyter notebooks, improvements
to the documentation, submitting bug report and feature requests, or even writing
code which can be incorporated into MetPy for everyone to use. If you get stuck at
any point you can create an [issue on GitHub](https://github.com/metpy/MetPy/issues)
or contact us at one of the other channels mentioned below.

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point.

## Ground Rules
The goal is to maintain a diverse community that's pleasant for everyone. Please
be considerate and respectful of others. Other items:

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
When creating a new issue, please be as specific as possible. Include the version
of the code you were using, as well as what operating system you are running.
If possible, include complete, minimal example code that reproduces the problem.

## Pull Requests
**Working on your first Pull Request?** You can learn how from this *free* series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/series/how-to-contribute-to-an-open-source-project-on-github)

We love pull requests from everyone. Fork, then clone the repo:

    git clone git@github.com:your-username/metpy.git

Install metpy:

    pip install .

Install py.test and make sure the tests pass:

    pip install pytest
    py.test

Make your change. Add tests for your change. Make the tests pass:

    py.test

Commit the changes you made. Chris Beams has written a [guide](http://chris.beams.io/posts/git-commit/) on how to write good commit messages.

Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/metpy/metpy/compare/

## Code Review
Once you've submitted a Pull Request, at this point you're waiting on us. You
should expect to hear at least a comment within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted:

* Write tests.
* Follow [PEP8][pep8] for style. (The `flake8` utility can help with this.)
* Write a [good commit message][commit].

Pull requests will automatically have tests run by Travis. This includes
running both the unit tests as well as the `flake8` code linter.

[pep8]: https://www.python.org/dev/peps/pep-0008/
[commit]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

## Other Channels
If you're interested in contacting us or being a part of the community in
other ways, feel free to contact us in
[MetPy's Gitter Channel](https://gitter.im/metpy/MetPy) or through Unidata's
[python-users](https://www.unidata.ucar.edu/support/#mailinglists) mailing list.
