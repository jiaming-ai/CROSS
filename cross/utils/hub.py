"""torch.hub loading that survives an unreachable github.

torch.hub.load("owner/repo", ...) probes github for the default branch on every call, even when the repo is cached, so a
github outage (e.g. HTTP 504) fails the run.  hub_load falls back to the cached checkout <hub dir>/<owner>_<repo>_main
on network errors, and uses it directly with CROSS_HUB_OFFLINE=1.  Weights come from torch.hub's checkpoint cache in
both cases (load_state_dict_from_url inside hubconf skips the download when the file exists)."""
import os
import warnings

import torch


def hub_offline() -> bool:
    return os.environ.get("CROSS_HUB_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


def hub_local_dir(repo: str) -> str:
    owner, name = repo.split("/")
    return os.path.join(torch.hub.get_dir(), f"{owner}_{name}_main")


def is_network_error(e: BaseException) -> bool:
    """URLError / HTTPError / socket timeouts / ConnectionError (all OSError), http.client errors, and the RuntimeError
    torch.hub raises from _parse_repo_info when there is "no internet connection"."""
    import http.client
    import urllib.error
    if isinstance(e, (urllib.error.URLError, OSError, http.client.HTTPException, TimeoutError)):
        return True
    if isinstance(e, RuntimeError):
        msg = str(e).lower()
        return ("no internet" in msg) or ("could not be found in the cache" in msg) or isinstance(e.__cause__, (urllib.error.URLError, OSError))
    return False


def hub_load(repo: str, entry: str, local_dir: str = None, force_local: bool = False, **kw):
    """torch.hub.load(repo, entry, **kw), or the cached checkout `local_dir` (default hub_local_dir(repo)) offline or
    when github is unreachable.  Errors other than network errors (e.g. inside hubconf) are re-raised unchanged."""
    local_dir = local_dir or hub_local_dir(repo)
    if (force_local or hub_offline()) and os.path.isdir(local_dir):
        return torch.hub.load(local_dir, entry, source="local", **kw)
    try:
        return torch.hub.load(repo, entry, trust_repo=True, **kw)
    except Exception as e:
        if not (is_network_error(e) and os.path.isdir(local_dir)):
            raise
        warnings.warn(f"torch.hub github load of {repo} failed with a network error ({type(e).__name__}: {e}); "
                      f"falling back to the local cache {local_dir}")
        return torch.hub.load(local_dir, entry, source="local", **kw)
