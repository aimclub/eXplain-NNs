Contribution
=================

We welcome you to [check the existing issues](https://github.com/Med-AI-Lab/eXplain-NNs/issues) for bugs or
enhancements to work on. If you have an idea for an extension to ECG,
please [file a new issue](https://github.com/Med-AI-Lab/eXplain-NNs/issues/new) so we can
discuss it.

Make sure to familiarize yourself with the project layout before making
any major contributions.


How to contribute
-----------------

The preferred way to contribute to ECG is to fork the [main repository](https://github.com/Med-AI-Lab/eXplain-NNs/) on GitHub:

1. Fork the [project repository](https://github.com/Med-AI-Lab/eXplain-NNs):
   click on the 'Fork' button near the top of the page. This creates a
   copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

         $ git clone git@github.com:YourUsername/ECG.git
         $ cd ECG

3. Create a branch to hold your changes:

         $ git checkout -b my-contribution

4. Make sure your local environment is setup correctly for development.
   Installation instructions are almost identical to [the user
   instructions](installing.md) except that ECG should *not* be
   installed. If you have ECG installed on your computer then make
   sure you are using a virtual environment that does not have ECG
   installed.

5. Start making changes on your newly created branch, remembering to
   never work on the ``main`` branch! Work on this copy on your
   computer using Git to do the version control.

6. To check your changes haven't broken any existing tests and to check
   new tests you've added pass run the following (note, you must have
   the ``pytest`` package installed within your dev environment for this
   to work):

         $ pytest tests/api-tests.py

7. When you're done editing and local testing, run:

         $ git add modified_files
         $ git commit

8. To record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

9. Finally, go to the web page of your fork of the ECG repo, and click
'Pull Request' (PR) to send your changes to the maintainers for review. <br/>
(If it looks confusing to you, then look up the [Git documentation](http://git-scm.com/documentation) on the web.)

Before submitting your pull request
-----------------------------------

Before you submit a pull request for your contribution, please work
through this checklist to make sure that you have done everything
necessary so we can efficiently review and accept your changes.

If your contribution changes ECG in any way:

-  Update the
   [documentation](https://github.com/Med-AI-Lab/eXplain-NNs/tree/main/docs)
   so all of your changes are reflected there.

-  Update the
   [README](https://github.com/Med-AI-Lab/eXplain-NNs/blob/main/README.md)
   if anything there has changed.

If your contribution involves any code changes:

-  Update the [project unit tests](https://github.com/Med-AI-Lab/eXplain-NNs/tree/main/tests) to test your code changes.

-  Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

-  Double-check that the new dependency is easy to install via ``pip``
   or Anaconda and supports Python 3.7. If the dependency requires a
   complicated installation, then we most likely won't merge your
   changes because we want to keep ECG Recognition Library easy to install.

-  Add the required version of the library to [requirements.txt](https://github.com/Med-AI-Lab/eXplain-NNs/blob/main/requirements.txt)

Contribute to the documentation
-------------------------------
Take care of the documentation. Add docstrings for API methods.

When introducting changes to API, update documentation with: `$ pdoc ECG/api.py -o ./docs`

After submitting your pull request
----------------------------------

Pull requests into `main` branch are required to pass status checks that are automatically run with a Github Actions action, that builds the project and runs tests.

Ensure that your pull request passes status checks before submitting it for a review.

Acknowledgements
----------------

This document guide is based at well-written [FEDOT Framework contribution guide](https://github.com/nccr-itmo/FEDOT/blob/main/docs/source/contribution.rst).