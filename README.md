# Gan's Utils

![PyPi Publish Badge](https://github.com/Michael-Tu/ganutils/workflows/Publish%20PyPi%20Package/badge.svg)

![Documentation Generation Badge](https://github.com/Michael-Tu/ganutils/workflows/Automated%20Docs%20Generation/badge.svg)

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

## Dependencies

You'll need `twine` installed via `pip` and PyPi credentials set at [`$HOME/.pypirc`](https://docs.python.org/3.3/distutils/packageindex.html#pypirc)
