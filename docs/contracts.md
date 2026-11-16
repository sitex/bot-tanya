# Behavior contracts

`create_app` serves existing `/v1/chat` plus `/v2/telegram/chat`. The additive v2
router can be imported into an existing app without migrating its v1 routes.
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
Only status, connect, list and disconnect map to callbacks; reading a message
requires explicit `/gmail read ID`. No mail content is ingested into AI history,
memory, summary or the journal. Previews and queries remain untrusted plain text.

Local menus, callback notices, native command descriptions and labels use the
English/Russian catalog. Translated labels preserve their opaque action IDs.
AI answers bypass translation. Gmail translates static headings inside its
adapter; the main renderer never translates returned previews, IDs or queries.

Limits: summaries can be wrong; a retrieval match is a candidate rather than proof
of relevance or contradiction. No minimum semantic relevance threshold is claimed.
The model and synthetic semantic smoke belong to `tanya-memory`; the bot's ordinary
offline tests do not download or benchmark a model. Gmail remains disconnected
until the owner completes the separate Google authorization and verification.
