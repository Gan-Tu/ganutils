# Gan's Utils

![Auto Release Badge](https://github.com/Michael-Tu/ganutils/workflows/Auto%20Release/badge.svg) ![PyPi Publish Badge](https://github.com/Michael-Tu/ganutils/workflows/Publish%20PyPi%20Package/badge.svg) ![Issues Labeler Badge](https://github.com/Michael-Tu/ganutils/workflows/Issues%20Labeler/badge.svg)

This is an installable python package for scripts I wrote for myself.

- PyPi Package Page [link](https://pypi.org/project/ganutils/)
- Documentation [link](https://michael-tu.github.io/ganutils/)

## Installation

To install, simply run

```
$ pip install ganutils
```

## Deployment

To deploy new versions to PyPi, change version number in `setup.py` and either

- create a new release for repo and GitHub workflow will automatically deploy it
- run `python setup.py sdist` and then `twine upload dist/*`.

## Update Documentation

Currently, upon each new release, my GitHub action will automatically build and generate documentation necessary.

However, to manually generate documentation, I use [pdoc3](https://pdoc3.github.io):

```
$ pip3 install pdoc3 # install dependency
$ pdoc3 --html ganutils # resulting docs will be in a new html/ folder
```

## Dependencies

You'll need `twine` installed via `pip` and PyPi credentials set at [`$HOME/.pypirc`](https://docs.python.org/3.3/distutils/packageindex.html#pypirc)
