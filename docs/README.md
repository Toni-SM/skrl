# Documentation

## Install Sphinx and Read the Docs Sphinx Theme

```bash
cd docs
pip install -r requirements.txt
```

## Building the documentation

```bash
cd docs
make html
```

Building each time a file is changed:

```bash
cd docs
sphinx-autobuild ./source/ _build/html
```

## Useful links

- [Sphinx directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
- [Math support in Sphinx](https://www.sphinx-doc.org/en/1.0/ext/math.html)
