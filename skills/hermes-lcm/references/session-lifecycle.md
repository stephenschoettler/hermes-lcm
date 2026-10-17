# Session lifecycle and rotate

Hermes `/new` starts a new host session for that session key. LCM then forgets that conversation's summary nodes (every depth, including the last finalized predecessor) and clears `last_finalized_session_id`, so the next compression cannot carry those summaries back in. Each session key is separate. A host that also resets another key, such as a forum General thread, must send its own reset; LCM forgets only the conversation id on that hook. Other conversations are left alone.

Raw messages for the old session stay in `lcm.db` and remain available through explicit recall. `/new` does not delete them. Compression boundaries are not `/new`: those may still carry summaries.

## `/lcm rotate`

`/lcm rotate` is different from `/new`:

- it keeps the current `session_id` and `conversation_id`;
- preview is read-only;
- apply creates/updates the rolling rotate backup first;
- it preserves the configured fresh tail;
- it advances the lifecycle frontier past older raw messages so bootstrap does not replay them into active context;
- it does not delete raw source rows or call a summarization model.

Run normal compaction before rotate when older material must be represented in summary nodes. Even without a summary, pre-tail raw rows remain recoverable through `lcm_load_session` and `lcm_expand`.

Rotate refuses ignored or stateless sessions. Repeating an already-satisfied rotate reports a no-op and preserves the previous known-good rolling backup.

Use a separate session when the user wants a new active conversational boundary. Use rotate when the problem is active transcript/frontier size without changing identity.
