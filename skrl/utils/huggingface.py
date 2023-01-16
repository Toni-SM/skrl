from skrl import logger, __version__


def download_model_from_huggingface(repo_id: str, filename: str = "agent.pt") -> str:
    """Download a model from Hugging Face Hub

    :param repo_id: Hugging Face user or organization name and a repo name separated by a ``/``
    :type repo_id: str
    :param filename: The name of the model file in the repo (default: ``"agent.pt"``)
    :type filename: str, optional

    :return: Local path of file or if networking is off, last version of file cached on disk
    :rtype: str
    """
    try:
        import huggingface_hub
    except ImportError:
        logger.error("Hugging Face Hub package is not installed. Use 'pip install huggingface-hub' to install it")
        huggingface_hub = None

    if huggingface_hub is None:
        raise ImportError("Hugging Face Hub package is not installed. Use 'pip install huggingface-hub' to install it")

    # download and cache the model from Hugging Face Hub
    downloaded_model_file = huggingface_hub.hf_hub_download(repo_id=repo_id,
                                                            filename=filename,
                                                            library_name="skrl",
                                                            library_version=__version__)

    return downloaded_model_file
