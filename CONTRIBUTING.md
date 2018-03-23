# Contributing

## Introduction
First off, thank you for considering contributing to MetPy. MetPy is community-driven
project, so it's people like you that make MetPy useful and successful.

Following these guidelines helps to communicate that you respect the time of the
developers managing and developing this open source project. In return, they
should reciprocate that respect in addressing your issue, assessing changes, and
helping you finalize your pull requests.

We love contributions from community members, just like you! There are many ways
to contribute, from writing tutorials or examples, improvements
to the documentation, submitting bug report and feature requests, or even writing
code which can be incorporated into MetPy for everyone to use. If you get stuck at
any point you can create an [issue on GitHub](https://github.com/Unidata/MetPy/issues)
or contact us at one of the other channels mentioned below.

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point. Also, checkout the [Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software contributions.

## Getting Started

Interested in helping build MetPy? Have code from your research that you believe others will
find useful? Have a few minutes to tackle an issue? In this guide we will get you setup and
integrated into contributing to MetPy!

## What Can I Do?
* Tackle any [issues](https://github.com/Unidata/MetPy/issues) you wish! We have a special
  label for issues that beginners might want to try. Have a look at our
  [current beginner issues.](https://github.com/unidata/metpy/issues?q=is%3Aopen+is%3Aissue+label%3A%22Difficulty%3A+Beginner%22)

* Contribute code you already have. It doesn’t need to be perfect! We will help you clean
  things up, test it, etc.

* Make a tutorial or example of how to do something.

## How Can I Talk to You?
Discussion of MetPy development often happens in the issue tracker and in pull requests.
In addition, the developers monitor the
[Gitter chat room](https://gitter.im/Unidata/MetPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
for the project as well.

## Ground Rules
The goal is to maintain a diverse community that's pleasant for everyone. Please
be considerate and respectful of others by following our [code of conduct](https://github.com/Unidata/MetPy/blob/master/CODE_OF_CONDUCT.md). Other items:

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
**Working on your first Pull Request?** You can learn how from this *free* video series [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github), Aaron Meurer's [tutorial on the git workflow](https://www.asmeurer.com/git-workflow/), or the guide [“How to Contribute to Open Source"](https://opensource.guide/how-to-contribute/).
We love pull requests from everyone. Fork, then clone the repo:

    git clone git@github.com:your-username/metpy.git

Install metpy:

    pip install .

Install py.test (at least version 2.4) and make sure the tests pass:

    pip install pytest
    py.test

Make your change. Add tests for your change. Make the tests pass:

    py.test

Commit the changes you made. Chris Beams has written a [guide](https://chris.beams.io/posts/git-commit/) on how to write good commit messages.

Push to your fork and [submit a pull request][pr].

[pr]: https://github.com/Unidata/metpy/compare/

For the Pull Request to be accepted, you need to agree to the
MetPy Contributor License Agreement (CLA). This will be handled automatically
upon submission of a Pull Request.
See [here](https://github.com/Unidata/MetPy/blob/master/CLA.md) for more
explanation and rationale behind MetPy's CLA.

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

[pep8]: http://pep8.org
[commit]: https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

## Other Channels
If you're interested in contacting us or being a part of the community in
other ways, feel free to contact us in
[MetPy's Gitter Channel](https://gitter.im/Unidata/MetPy), ask questions using the
["metpy" tag on Stack Overflow](https://stackoverflow.com/questions/tagged/metpy),
email [Unidata's Python support address](mailto:support-python@unidata.ucar.edu),
or through Unidata's
[python-users](https://www.unidata.ucar.edu/support/#mailinglists) mailing list.
