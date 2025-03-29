## GitHub Actions

### Relevant links

- `runs-on`:
  - [Standard GitHub-hosted runners for public repositories](https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories)
  - [GitHub Actions Runner Images](https://github.com/actions/runner-images)
- `actions/setup-python`:
  - [Building and testing Python](https://docs.github.com/en/actions/use-cases-and-examples/building-and-testing/building-and-testing-python)
  - [Available Python versions](https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json)

### Run GitHub Actions locally with nektos/act

[nektos/act](https://nektosact.com/) is a tool to run GitHub Actions locally. Install it as a [GitHub CLI](https://cli.github.com/) extension via [this steps](https://nektosact.com/installation/gh.html).

#### Useful commands

* List workflows/jobs:

  ```bash
  gh act -l
  ```

* Run a specific job:

  Use `--env DELETE_HOSTED_TOOL_PYTHON_CACHE=1` to delete the Python cache.

  ```bash
  gh act -j Job-ID
  ```
