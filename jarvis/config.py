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
from typing import Any
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
def _read_roots_from_env() -> tuple[Path, ...]:
    """The read scope, from `JARVIS_READ_ROOTS` (comma-separated) or `JARVIS_READ_ROOT`.

    Two names because a single root is the common case and `READ_ROOT` reads better
    for it, while a scope is naturally a list. Both expand, both resolve, and an
    empty value falls back to the home directory rather than to an empty scope --
    an empty read scope is a safe failure, and silently making one would look like
    "the agent is broken" rather than "you configured it wrong".
    """
    multi = _env("JARVIS_READ_ROOTS")
    if multi:
        parts = [p.strip() for p in multi.split(",") if p.strip()]
    else:
        parts = [_env("JARVIS_READ_ROOT", str(Path.home()))]
    return tuple(Path(p).expanduser().resolve() for p in parts)


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
# Read NEVER needs these. Each entry is either a credential, a kernel-internal
# pseudo-filesystem, or another agent's private store -- and the kernel ones matter
# because "search the whole system" naturally reaches /proc, where every process's
# environment and open file descriptors are readable. `/proc/self/environ` is the
# API key.
#
# Widened deliberately BEFORE the scope was widened, per the plan. Adding paths to
# the read scope while these are missing is how an agent ends up reporting the
# contents of someone's browser profile.
DENY_GLOBS: tuple[str, ...] = (
    # --- credentials and keys
    "~/.ssh/*", "~/.aws/*", "~/.gnupg/*", "~/.kube/*", "~/.config/gcloud/*",
    "~/.docker/config.json", "~/.npmrc", "~/.pypirc", "~/.netrc",
    "~/.config/gh/*", "~/.config/hub", "~/.local/share/keyrings/*",
    "~/.local/share/kwallet*", "~/.password-store/*",
    "*/id_rsa*", "*/id_ed25519*", "*/id_ecdsa*", "*.pem", "*.key", "*_rsa",
    "*/.env", "*/.env.*", "*/.netrc", "*/credentials*", "*api_key*", "*apikey*",
    "*.p12", "*.keystore", "*/.git-credentials", "*/.bash_history", "*/.zsh_history",
    "*/.pgpass", "*/.my.cnf", "*/.gem/credentials",
    # --- kernel and device pseudo-filesystems. Reading these is not a privacy
    # question so much as a correctness one: /proc reports live state that changes
    # between two reads of the same path, so any "file" found there is a fiction.
    "/proc/*", "/sys/*", "/dev/*", "/run/secrets/*",
    # --- browser profiles. Cookies and session storage are a live account takeover.
    "*/.mozilla/firefox/*", "*/.config/google-chrome/*", "*/.config/chromium/*",
    "*/.config/BraveSoftware/*", "*/Library/Application Support/Google/Chrome/*",
    "*/.config/opera/*", "*/.config/microsoft-edge/*",
    # --- other agents' and tools' private state
    "~/.opencode/*", "~/.claude/*", "~/.config/opencode/*", "~/.ollama/*",
    "~/.local/share/Trash/*", "~/.local/share/recently-used.xbel",
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
    # Fallbacks are tried in order when a call fails transiently. The comment that
    # used to sit here -- that Groq's daily limit is per-model, so a fallback must
    # stay on the same model -- was about one provider and no longer described what
    # the code does: this is a generic list and a caller may legitimately span
    # providers. What still holds is that a fallback is only useful if the endpoint
    # can serve it, which `preflight` checks before any task runs.
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
    executor_max_calls: int = 12
    executor_timeout_s: int = 30
    planner_cot: bool = True
    verifier_cot: bool = False
    verifier_rubric: bool = True
    use_rg: bool | None = None
    # None -> the react loop derives its turn cap from budget.max_iterations, which
    # is the real authority. Set this only to make it smaller.
    react_max_turns: int | None = None

    def replaced(self, **kw) -> "Settings":
        """Copy with overrides. Tests use this instead of mutating globals."""
        return replace(self, **kw)


def _limits_from_env() -> Limits:
    """Every `Limits` field from `LIMIT_<FIELD>`, so no limit is hardcoded.

    Driven off the dataclass rather than written out, because a hand-written list
    drifts: adding a field to `Limits` and forgetting this function is how you get a
    limit that exists in the source and cannot be configured. Fields whose declared
    type is not a number (or bool) are skipped rather than mis-parsed.

    Reads `LIMIT_FOO` and, as a convenience, the unprefixed `FOO`, because the
    unprefixed name is what someone types and the prefixed one is what the docs say.
    """
    import dataclasses

    kw: dict[str, Any] = {}
    for f in dataclasses.fields(Limits):
        want_bool = f.type in ("bool", bool)
        raw = _env(f"LIMIT_{f.name.upper()}") or _env(f.name.upper())
        if raw is None:
            continue
        try:
            if want_bool:
                kw[f.name] = raw.strip().lower() not in ("0", "false", "no", "off")
            elif f.type in ("float", float):
                kw[f.name] = float(raw)
            else:
                kw[f.name] = int(raw)
        except ValueError:
            # A malformed value must not crash startup -- it must be visible. The
            # default stands and the caller finds out from behaviour, which is worse
            # than a message, so note it rather than swallowing it.
            print(f"[config] ignoring malformed LIMIT_{f.name.upper()}={raw!r}")
    return Limits(**kw)


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
        max_parallel_plans=int(_env("JARVIS_MAX_PARALLEL_PLANS", "4")),
    )
    # The key name is a compatibility question, not a preference: both were in use
    # at different points, and silently preferring the wrong one gives "no API key"
    # on a machine that has one.
    api_key = (_env("LLM_API_KEY") or _env("GROQ_API_KEY")
               or _env("OPENAI_API_KEY") or _env("OPENROUTER_API_KEY"))
    env_llm = LLMConfig(
        base_url=_env("LLM_BASE_URL") or _env("OPENAI_BASE_URL"),
        model=_env("LLM_MODEL"),
        dspy_model=_env("DSPY_MODEL"),
        api_key=api_key,
        timeout_s=int(_env("LLM_TIMEOUT_S", "60")),
        temperature=float(_env("LLM_TEMPERATURE", "0.0")),
        max_tokens=int(_env("LLM_MAX_TOKENS", "1500")),
        fallback_models=tuple(
            m.strip() for m in (_env("LLM_FALLBACK_MODELS") or "").split(",") if m.strip()),
    )
    settings = Settings(
        limits=_limits_from_env(),
        budget=env_budget,
        llm=env_llm,
        read_roots=_read_roots_from_env(),
        write_root=Path(_env("WRITE_WORKSPACE", "~/jarvis_workspace")).expanduser().resolve(),
        executor_enabled_tier=int(_env("EXECUTOR_ENABLED_TIER", "0")),
        executor_dry_run=env_flag("EXECUTOR_DRY_RUN", True),
        executor_max_calls=int(_env("EXECUTOR_MAX_CALLS", "12")),
        executor_timeout_s=int(_env("EXECUTOR_TIMEOUT_S", "30")),
        planner_cot=env_flag("PLANNER_COT", True),
        verifier_cot=env_flag("VERIFY_COT", False),
        verifier_rubric=env_flag("VERIFIER_RUBRIC", True),
        use_rg=None if _env("USE_RG") is None else env_flag("USE_RG", True),
        react_max_turns=int(_env("REACT_MAX_TURNS", "0")) or None,
    )
    return settings.replaced(**overrides) if overrides else settings


SETTINGS = load_settings()