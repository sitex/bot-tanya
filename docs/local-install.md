# Local installation

The API package requires Python 3.12 or newer. The optional local embedding bot
runtime is tested on Linux with Python 3.12.11, PyTorch 2.9.1+cpu and Sentence
Transformers 5.7.0. Importing `tanya_bot.turn_api.turn_router` does not import Torch
or Sentence Transformers. The API can run in a separate Python 3.13 environment
without the local extra. Keep one API writer and one bot runtime per private state
directory.

Install reviewed wheels for all four packages, with release SHA256 verification.
Use an isolated bot environment; do not replace an existing API environment.
For example, when all wheels are in a reviewed `release/` directory:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python 'torch==2.9.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --find-links release 'bot-tanya[local]==0.1.0'
```

The CPU index is explicit in the standalone development configuration. Ordinary
wheel installers must use the two commands above or explicitly provide that
index; dependency metadata cannot configure an installer's indexes. The local
extra requests the CPU build on Linux, and `tanya-memory[local]` pins Sentence
Transformers. Installing the base `bot-tanya` alone does not require either.
For a separate existing API environment, install only the base reviewed wheels.

Provision these external values before running `tanya-bot`:

| Setting | Meaning |
| --- | --- |
| `TANYA_DATA_DIR` | Bot state, identity, SDK session and audio directory |
| `TANYA_API_STATE_DIR` | Same private receipt directory used by the local companion API |
| `TANYA_CLIENT_TOKEN` or `TANYA_CLIENT_TOKEN_FILE` | Shared companion API authorization |
| `OPENAI_API_KEY` or `TANYA_API_KEY_FILE` | Provider credential; never place it in source |
| `TANYA_MODEL` | Explicit provider model; existing legacy aliases remain accepted |
| `TELEGRAM_API_ID`, `TELEGRAM_API_HASH` | Externally supplied Telethon application credentials |
| `TANYA_BOT_TOKEN` or a `bot-token` file in the bot directory | Provisioned bot token |
| `TANYA_MEMORY_MODEL_DIR` | Optional prepared, immutable, verified local model snapshot |
| `TANYA_GMAIL_CONFIG` | External private Gmail JSON configuration file |
| `TANYA_GMAIL_TOKEN_DIR` | External private Gmail credential directory |

The bot directory requires an externally provisioned `creation.json` with a
completed `stage`, the authorized `owner_id`, `bot_id` and `username`. Runtime
never creates a bot, changes the owner or imports another bot's identity.
Private directories use 0700; state and credential files use 0600. Synthetic test
identities are not usable production configuration. The companion endpoint is
loopback HTTP on port 8758; start the standalone app with
`uvicorn tanya_bot.app:app --host 127.0.0.1 --port 8758`, or include `turn_router`
in the existing compatible API process. Do not run two API writers on the same
receipt directory.

Without the model directory or local extra, retrieval reports lexical mode.
Prepare a local snapshot using `tanya-memory-prepare` according to that package's
README. The packaged manifest pins the model revision and ten file hashes.
Runtime verifies it before CPU loading, with no remote code or model downloads.
Use a bounded CPU configuration, such as `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2` and `TOKENIZERS_PARALLELISM=false`. Startup assembly runs model
loading in a worker. Retrieval and maintenance share a one-worker limiter outside
the controller state lock. Inference failure produces a safe diagnostic and
lexical retrieval. Corrupt disposable SQLite indexes are preserved and rebuilt
in a new private child directory; authoritative bot state is never reset.

Gmail configuration uses the `tanya-gmail` configuration schema: expected primary
email, desktop client ID and external client secret where applicable. Google
authorization is an explicit host-browser operation performed separately. Bot
connect never opens a browser. Status does not restore or verify credentials.
No message access occurs until the adapter verifies the exact configured account.
Disconnect removes local credentials without revoking remotely.

For offline development from the staging package, use existing sibling tools:

```bash
../.venv/bin/ruff check src tests
../.venv/bin/ruff format --check src tests
../.venv/bin/basedpyright --pythonpath ../.venv/bin/python
../.venv/bin/pytest -q
uv build --offline --out-dir dist
```

The suite blocks real socket connections. Telegram, provider and Gmail boundaries
are synthetic. Real authorization, publication and service installation are
separate deployment work. Back up private state before switching binaries;
rollback must preserve new receipts and use a compatible binary rather than
overwriting newer state with an old backup.
