"""ensure_jax_compilation_cache configures JAX, not only the environment."""

import os
import subprocess
import sys


def _run(program: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "JAX_COMPILATION_CACHE_DIR"}
    completed = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return completed.stdout.strip()


def test_existing_directory_set_after_jax_import_is_applied(tmp_path) -> None:
    # JAX reads JAX_COMPILATION_CACHE_DIR only at import; an early return that
    # only echoed the variable left the cache off (config value None).
    program = (
        "import os, jax\n"
        f"os.environ['JAX_COMPILATION_CACHE_DIR'] = {str(tmp_path)!r}\n"
        "from jaxstro.jaxconfig import ensure_jax_compilation_cache\n"
        "ensure_jax_compilation_cache()\n"
        "print(jax.config.jax_compilation_cache_dir)"
    )
    assert _run(program) == str(tmp_path)


def test_default_path_creates_and_applies_the_directory(tmp_path) -> None:
    program = (
        "import jax\n"
        "from jaxstro.jaxconfig import ensure_jax_compilation_cache\n"
        f"path = ensure_jax_compilation_cache(base_dir={str(tmp_path)!r})\n"
        "print(path == jax.config.jax_compilation_cache_dir)"
    )
    assert _run(program) == "True"
    assert (tmp_path / "xla_compilation_cache").is_dir()
