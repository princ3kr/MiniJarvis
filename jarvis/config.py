"""Every tunable constant in the system, in one place.

Nothing else in the package reads os.environ or hard-codes a limit. That is the whole
point of this module: the notebook spread config across a dozen cells, so a value
like MAX_FILE_BYTES had to be hunted for, and tests could not override it without
exec'ing cells.

Precedence (low to high):
    dataclass defaults  ->  .env in the project root  ->  real environment

`.env` is read rather than required because the notebook previously only worked if
you remembered to `export` three variables before starting Jupyter. Real env still
wins, so a shell override beats the file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader. Tolerates the `KEY = "value"` spacing and
    surrounding quotes that a shell `source` would reject."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        key = key.strip()
        if key:
            out[key] = value
    return out


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or _parse_env_file(PROJECT_ROOT / ".env").get(name) or default


def env_flag(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


# --------------------------------------------------------------------------- jail
# The read scope. A single list, used by every tool, every command operand, and the
# metadata index. The notebook had two independent jails (`safe_resolve` and
# `_resolve_in`), which disagreed about relative paths.
READ_ROOTS: tuple[Path, ...] = (
    Path(_env("JARVIS_READ_ROOT", str(Path.home()))).expanduser().resolve(),
)

# Where write-tier commands may touch. Frozen tier 1/2 lives here.
WRITE_ROOT: Path = Path(_env("WRITE_WORKSPACE", "~/jarvis_workspace")).expanduser().resolve()

# Directories that are never worth walking. Cheap `find` pruning, and the hint the
# agent is given so it does not report "not found" for something inside one of these.
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ipynb_checkpoints", "dist", "build",
    ".vscode", "site-packages",
    ".cache", ".config", ".cargo", ".dotnet", ".nv", ".npm", ".snap", ".var",
    ".triton", ".gnupg", ".ipython", ".jupyter", ".icons", ".local",
    ".mozilla", ".thunderbird", "scikit_learn_data",
})

# Deny-list. Wins over the read scope, so a tier-0 read cannot reach a secret even
# though the file is inside READ_ROOTS. `cat ~/.ssh/id_rsa` is read-only, so a tier
# rule alone would wave it through.
#
# Matching rules (see access.is_denied):
#   * a pattern containing "/" is matched against the whole absolute path, where
#     `*` crosses directory separators -- so `*/.env` blocks it at ANY depth
#   * a pattern without "/" is matched against the basename
#   * a leading "~" is expanded
DENY_GLOBS: tuple[str, ...] = (
    "~/.ssh/*", "~/.aws/*", "~/.gnupg/*", "~/.config/gcloud/*", "~/.kube/*",
    "*/id_rsa*", "*/id_ed25519*", "*/id_ecdsa*", "*.pem", "*.key", "*_rsa",
    "*/.env", "*/.env.*", "*/.netrc", "*/credentials*", "*api_key*", "*apikey*",
    "*.p12", "*.keystore", "*/.git-credentials", "*/.bash_history", "*/.zsh_history",
)

# --------------------------------------------------------------------------- files
# Blocks read_file from decoding these as text. NOT a search filter: a .zip is still
# findable and listable, it just cannot be read as text.
NO_TEXT_EXT: frozenset[str] = frozenset({
    ".pyc", ".so", ".dylib", ".dll", ".exe", ".o", ".a", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff",
    ".mp4", ".mov", ".mp3", ".wav", ".avi", ".mkv", ".webm",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".whl", ".rar", ".jar",
    ".db", ".sqlite", ".sqlite3", ".parquet", ".pkl", ".npy", ".npz",
    ".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5",
})

LISTABLE_ARCHIVE_EXT: frozenset[str] = frozenset(
    {".zip": "zipfile", ".tar": "tarfile", ".gz": "tarfile",
     ".tgz": "tarfile", ".bz2": "tarfile", ".xz": "tarfile"}
)

CATEGORIES: dict[str, frozenset[str]] = {
    "image": frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
                        ".tiff", ".svg", ".heic", ".avif"}),
    "video": frozenset({".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v"}),
    "audio": frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma"}),
    "document": frozenset({".pdf", ".doc", ".docx", ".odt", ".rtf", ".txt", ".md",
                           ".tex", ".epub", ".ppt", ".pptx", ".xls", ".xlsx"}),
    "spreadsheet": frozenset({".csv", ".tsv", ".xls", ".xlsx", ".ods", ".parquet"}),
    "notebook": frozenset({".ipynb"}),
    "archive": frozenset({".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z",
                          ".rar", ".jar"}),
    "code": frozenset({".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".h",
                       ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".pl",
                       ".lua", ".r", ".jl", ".swift", ".kt", ".scala", ".hs", ".ml",
                       ".ex", ".exs", ".sh", ".bash", ".zsh", ".sql", ".html",
                       ".css", ".vue", ".svelte"}),
    "data": frozenset({".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml"}),
    "binary": frozenset({".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".pyc", ".wasm"}),
}


@dataclass(frozen=True)
class Limits:
    """Per-operation caps. Defaults here, overridden per TaskContext."""

    # find
    find_max_results: int = 200
    dir_hint_max: int = 40
    fuzzy_pool_cap: int = 4_000
    fuzzy_cutoff: float = 0.60
    # Fuzzy needs ENOUGH characters to mean anything. difflib scores short strings on
    # incidental overlap: "search" vs "src" = 0.667 (just 's'+'rc'), "def" vs "ide" =
    # 0.667. Below this length a ratio is noise, and it lets instruction prose like
    # "Search the codebase" fuzzy-match every src/ folder on disk.
    fuzzy_min_len: int = 4
    near_miss_min: float = 0.15
    near_miss_max: int = 3
    # grep / read
    grep_max_results: int = 20
    grep_max_per_file: int = 3
    read_max_bytes: int = 50_000
    read_max_lines: int = 200
    read_context: int = 25
    binary_sniff: int = 8_192
    max_archive_members: int = 60
    max_results: int = 20
    # subprocess
    timeout_s: int = 15
    # runner
    exec_timeout_s: int = 60
    exec_max_output_bytes: int = 20_000
    cpu_rlimit_s: int = 60
    address_space_bytes: int = 1 << 30      # 1 GiB
    file_size_bytes: int = 1 << 22          # 4 MiB per file


@dataclass(frozen=True)
class Budget:
    """Ceilings for one task. Exhausting any of them ends the task honestly.

    These exist because a model-driven loop is unbounded by construction: it will
    keep calling find until something stops it, and tokens are metered elsewhere.
    """

    max_llm_calls: int = 12
    max_tool_calls: int = 40
    max_iterations: int = 8
    max_tokens: int = 200_000
    wall_clock_s: float = 300.0
    max_replans: int = 2
    max_parallel_plans: int = 4


@dataclass(frozen=True)
class LLMConfig:
    """Provider wiring. The interface is provider-shaped on purpose: whole-machine
    reads mean paths and snippets leave the machine, so `base_url` + egress policy
    are config, not a constant buried in a cell."""

    base_url: str | None = None
    model: str | None = None
    dspy_model: str | None = None
    api_key: str | None = None
    timeout_s: int = 60
    temperature: float = 0.0
    max_tokens: int = 900
    # Groq's daily token limit is per-model, so a fallback must stay on the SAME
    # model or the retry hits a different bucket that may also be exhausted.
    fallback_models: tuple[str, ...] = ()

    def with_fallbacks(self) -> "LLMConfig":
        return self


@dataclass(frozen=True)
class Settings:
    limits: Limits = field(default_factory=Limits)
    budget: Budget = field(default_factory=Budget)
    llm: LLMConfig = field(default_factory=LLMConfig)
    read_roots: tuple[Path, ...] = READ_ROOTS
    write_root: Path = WRITE_ROOT
    deny_globs: tuple[str, ...] = DENY_GLOBS
    skip_dirs: frozenset[str] = SKIP_DIRS
    # Frozen behind these flags. The discovery agent's manifest is max_tier=0, so
    # nothing below tier 0 is reachable by an agent even if a flag is flipped.
    executor_enabled_tier: int = 0
    executor_dry_run: bool = True
    planner_cot: bool = True
    verifier_cot: bool = False
    verifier_rubric: bool = True
    use_rg: bool | None = None

    def replaced(self, **kw) -> "Settings":
        """Copy with overrides. Tests use this instead of mutating globals."""
        return replace(self, **kw)


def load_settings(**overrides) -> Settings:
    """Build Settings from the environment, then apply explicit overrides.

    Explicit overrides win over env, which wins over defaults -- so a test can pin a
    value even if the developer's shell says otherwise.
    """
    env_budget = Budget(
        max_llm_calls=int(_env("JARVIS_MAX_LLM_CALLS", "12")),
        max_tool_calls=int(_env("JARVIS_MAX_TOOL_CALLS", "40")),
        max_iterations=int(_env("JARVIS_MAX_ITERATIONS", "8")),
        max_tokens=int(_env("JARVIS_MAX_TOKENS", "200000")),
        wall_clock_s=float(_env("JARVIS_WALL_CLOCK_S", "300")),
        max_replans=int(_env("JARVIS_MAX_REPLANS", "2")),
    )
    env_llm = LLMConfig(
        base_url=_env("LLM_BASE_URL"),
        model=_env("LLM_MODEL"),
        dspy_model=_env("DSPY_MODEL"),
        api_key=_env("GROQ_API_KEY") or _env("LLM_API_KEY"),
        timeout_s=int(_env("LLM_TIMEOUT_S", "60")),
    )
    settings = Settings(
        budget=env_budget,
        llm=env_llm,
        executor_enabled_tier=int(_env("EXECUTOR_ENABLED_TIER", "0")),
        executor_dry_run=env_flag("EXECUTOR_DRY_RUN", True),
        planner_cot=env_flag("PLANNER_COT", True),
        verifier_cot=env_flag("VERIFY_COT", False),
        verifier_rubric=env_flag("VERIFIER_RUBRIC", True),
        use_rg=None if _env("JARVIS_USE_RG") is None
        else env_flag("JARVIS_USE_RG", True),
    )
    return settings.replaced(**overrides) if overrides else settings


SETTINGS = load_settings()