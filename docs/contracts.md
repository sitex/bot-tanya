# Behavior contracts

`create_app` serves `/v1/chat`, `/v2/telegram/chat`, and the additive
`/v3/telegram/dialogue/phase` route. Existing apps can include `turn_router` and
`dialogue_router` without migrating v1 routes.
Version-one serialization, default values and receipt fingerprints remain
unchanged. Version-two reservations freeze `ChatRequest` and `TurnContext` before
the provider send, and use a separate `turns-v2` ledger. One structured provider
response carries text, explicit memory updates, summary and ephemeral delivery.
They become recoverable together in one SQLite commit. There are no repair calls,
automatic UNKNOWN retries or paid background summaries.

The gateway reads the core's atomic snapshot to select its frozen protocol.
Recovery reads SQLite only, never HTTP, generation or speech synthesis. It works
while the controller already holds its state transaction. Missing reservations,
changed input, unsafe files and corrupt authoritative state fail closed.
Legacy explicit cached recovery remains supported. Enhanced cancelled, archived
or source-stale results cannot update history, memory, summary or delivery plans.
Source-stale receipts are not automatically delivered; archived receipts stay
outside the active conversation. Explicit cached recovery of a cancelled enhanced
answer may display its text without applying new derived data or generating audio.

Manual memory/settings mutations advance the context epoch and revision. Memory
removal or correction invalidates summaries and excludes old history through a
persisted barrier without rewriting receipts or the journal. Retrieval always
receives current sources, so cached vectors cannot resurrect deleted entries.
Recent history, retrieved memories, persona, language, current input and summary
share the 16000-character budget. Summary input yields to the current message
when necessary; a partial summary chain cannot advance derived coverage, so the
prior summary is retained with a coverage-gap mark. Summaries are fallible context,
not instructions or authority for new durable facts.

`/new` requires a saved current confirmation bound to message and token. It resets
history and summary, advances the epoch, and preserves settings, memory, journal
and still-valid daily maintenance metadata. Old holds remain archived. Daily
maintenance runs at startup and every hour, marks at most one successful pass per
UTC date, retries failed/stale passes later, and catches up once after downtime.
CPU work happens outside the state lock. Applying results checks source revisions;
committing maintenance metadata alone does not invalidate a prepared chat.

Delivery is bounded by `tanya-persona` and applies to one answer. New voice
reservations persist the selected plan; the same plan supplies fixed speech
instructions and speed to TTS. Reservations without a plan retain the exact old
speech digest. Cancellation and UNKNOWN holds never authorize regeneration.

Only the configured owner in a private conversation may invoke Gmail. The bot
durably claims the update before releasing its state lock for the one account
adapter. Duplicate updates cannot repeat an action. A returned result is shown
only if its original conversation remains active. Static Gmail callbacks are
checked against saved reply tokens, Telegram message IDs and conversation IDs.
Only status, connect, list and disconnect map to callbacks. `/gmail read ID`
returns a plain preview. Explicit read/list gestures may restore saved credentials
and reverify the exact configured account, without opening browser authorization.
Status remains local and does not prove a saved grant is currently valid.

New installed dialogue turns use a separately marked v3 reservation. The first
model phase either answers normally or proposes one readonly search/read tool.
The bot executes the proposed tool for the authenticated owner; the API cannot
create mailbox authority. Search reads one page of up to five previews; selected
read accepts up to three stable references. A mail question uses at most two model
calls; an ordinary answer uses one. The final phase has no tools. There is no
automatic pagination, attachment download, URL fetch, sending or modification.
Mail is read only for a current owner request or a clear selection follow-up.

Each model phase and Google sub-operation is durably reserved before dispatch.
The exact model payload and completed result are saved. Cancellation fences
prevent later phases. Recovery reads receipts only; a cached tool proposal never
starts a search or a final model phase. Missing/UNKNOWN authority is never rebuilt
by repeating an external operation. V3 routing markers prevent fallback to v2.
The API checks current core cancellation and owner authority atomically with its
dispatch claim. Every nested acquisition takes core before sidecar; both locks
are released before external I/O. A core cancellation remains effective even if
the process dies before writing its sidecar fence.

Mail data is untrusted provider input. Mail-tainted turns have no automatic memory
updates, retain the clean general summary with a coverage gap, and exclude their
history entries from later automatic context and memory preparation. Requested
answers remain in Telegram and normal response receipts. Explicit owner actions
such as saving an answer remain separate from automatic memory.

Selections expire after 30 idle minutes or two hours total. References are checked
again after the decision and before every selected read. Source-bearing turn
snapshots are pruned after 24 hours on access and hourly maintenance; routing,
UNKNOWN tombstones and final answers remain. The ledger allows at most 1024 roots
and 1 MiB per root; it refuses new work instead of evicting replay protection.
Preview data is capped at 16000 characters, the entire provider payload at 32000,
and each provider response at 256 KiB. Ordinary runtime context reserves 8000
characters for history, summary, persona and memories, yielding to the latest
owner input when necessary. Very large input can still exceed the combined mail
envelope and is rejected without a repair generation.

Core replies reserve capacity for every unresolved request before new chat calls
or local mutations. Missing state, missing receipt databases/tables and ambiguous
partial storage fail closed. Valid legacy stores are adopted without rewriting
their bytes. Keep continuity lock files with backups; whole-directory deletion
and rollback of all authority and witnesses remain outside local detection.

Local menus, callback notices, native command descriptions and labels use the
English/Russian catalog. Translated labels preserve their opaque action IDs.
AI answers bypass translation. Gmail translates static headings inside its
adapter; the main renderer never translates returned previews, IDs or queries.

Limits: summaries can be wrong; a retrieval match is a candidate rather than proof
of relevance or contradiction. No minimum semantic relevance threshold is claimed.
The model and synthetic semantic smoke belong to `tanya-memory`; the bot's ordinary
offline tests do not download or benchmark a model. Gmail remains disconnected
until the owner completes the separate Google authorization and verification.
