# Tanya

A personal Telegram companion composed of independently reusable Python packages.
English is the default interface language; Russian is available in Settings.

With a separately authorized personal Gmail account, the owner can ask about
recent mail in ordinary conversation, search messages and follow up on stable
message references. Reading is readonly; reply drafts stay in chat. Mail sources
are kept separate from automatic personal memories. No browser login occurs at
bot startup.

This project is an independent implementation. It contains no Tolan application
assets, private prompts, personal conversations, credentials, or subscription API.

## Components

| Repository | Responsibility |
| --- | --- |
| [bot-tanya](https://github.com/sitex/bot-tanya) | Telegram, conversation orchestration, provider, speech, receipts and localization |
| [tanya-memory](https://github.com/sitex/tanya-memory) | Local semantic retrieval, derived summaries and safe maintenance |
| [tanya-persona](https://github.com/sitex/tanya-persona) | Editable character and bounded per-reply delivery |
| [tanya-gmail](https://github.com/sitex/tanya-gmail) | Explicit, personal-account-only, read-only Gmail access |

The base package keeps API orchestration light. Local CPU inference is an optional
extra; it never downloads model files during bot operation. Existing version-one
API defaults and receipts remain compatible. Fresh Telegram state uses Tanya and
English; stored names, voices and languages are preserved, and legacy snapshots
without a language retain Russian.

Clone this repository and run `uv sync --locked` with Python 3.12 to install
the bot and its three component repositories at immutable Git revisions.
For Linux CPU inference, use `uv sync --locked --extra local` and prepare
the separately supplied model files as described in tanya-memory.
No account credentials are included; provision the external settings first.

See [local installation](docs/local-install.md) for separate API and bot runtimes,
external private configuration and offline checks, and
[behavior contracts](docs/contracts.md) for recovery and privacy boundaries.
