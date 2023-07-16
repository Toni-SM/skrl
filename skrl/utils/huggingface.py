from skrl import __version__, logger


def download_model_from_huggingface(repo_id: str, filename: str = "agent.pt") -> str:
    """Download a model from Hugging Face Hub

    :param repo_id: Hugging Face user or organization name and a repo name separated by a ``/``
    :type repo_id: str
    :param filename: The name of the model file in the repo (default: ``"agent.pt"``)
    :type filename: str, optional

    :raises ImportError: The Hugging Face Hub package (huggingface-hub) is not installed
    :raises huggingface_hub.utils._errors.HfHubHTTPError: Any HTTP error raised in Hugging Face Hub

    :return: Local path of file or if networking is off, last version of file cached on disk
    :rtype: str

    Example::

        # download trained agent from the skrl organization (https://huggingface.co/skrl)
        >>> from skrl.utils.huggingface import download_model_from_huggingface
        >>> download_model_from_huggingface("skrl/OmniIsaacGymEnvs-Cartpole-PPO")
        '/home/user/.cache/huggingface/hub/models--skrl--OmniIsaacGymEnvs-Cartpole-PPO/snapshots/892e629903de6bf3ef102ae760406a5dd0f6f873/agent.pt'

        # download model (e.g. "policy.pth") from another user/organization (e.g. "org/ddpg-Pendulum-v1")
        >>> from skrl.utils.huggingface import download_model_from_huggingface
        >>> download_model_from_huggingface("org/ddpg-Pendulum-v1", "policy.pth")
        '/home/user/.cache/huggingface/hub/models--org--ddpg-Pendulum-v1/snapshots/b44ee96f93ff2e296156b002a2ca4646e197ba32/policy.pth'
    """
    logger.info(f"Downloading model from Hugging Face Hub: {repo_id}/{filename}")
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
