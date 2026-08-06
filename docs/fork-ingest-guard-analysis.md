# Fork/Side-Channel Ingest Guard — Risk Analysis & Test Plan

## Problem

A forked agent (background review, cron side-channel) shares the parent's
session_id but carries a different, shorter message list. When it calls
`should_compress_preflight()` → `_ingest_messages()`, the short list
overwrites the stored tail. The parent's next reconcile fails suffix
matching → cursor=0 → full re-ingest → duplication.

## Proposed Fix

In `_reconcile_ingest_cursor_from_store`, before the final `cursor=0`
fallback ("persisted ambiguous delta"), add a guard:

```python
if (
    cursor is None
    and len(incoming_identities) < session_count
    and not set(
        incoming_identities[-min(5, len(incoming_identities)):]
    ).intersection(set(stored_tail))
):
    # Short batch with no tail overlap: fork or side-channel.
    # A legitimate restart always shares tail messages with the
    # durable store. Skip the batch to protect the stored tail.
    return len(messages)
```

## Risk Assessment

### False positive: legitimate ingest skipped

| Scenario | Risk | Why safe |
|---|---|---|
| Normal restart, full replay | None | incoming >= session_count → guard skipped |
| Normal restart, short delta | None | delta tail overlaps stored tail → intersection non-empty |
| Gateway restart, stale snapshot | None | already caught by `_is_suspicious_stale_no_overlap_snapshot` upstream |
| Very short session (≤5 msgs) + restart | Low | `len(incoming) < session_count` may be false if all msgs present |
| Session with all messages compacted | Medium | session_count could be 0 → guard skipped (session_count=0 returns early) |
| Ignore-pattern filtering removes overlap | Medium | incoming_identities already filters ignored patterns; stored_tail also filters → consistent |

### False negative: fork ingest not caught

| Scenario | Risk | Mitigation |
|---|---|---|
| Fork carries parent's last 5+ messages verbatim | Medium | Intersection non-empty → guard skipped → duplication possible. But this means the fork IS replaying parent context, so re-ingest is less harmful (content already stored). |
| Fork has exactly session_count messages | Low | `len < session_count` false → guard skipped. Unlikely for forks (they carry shorter lists). |

### Performance

- `set()` construction on tail identities: O(n) where n = min(5, len) + len(stored_tail)
- stored_tail already materialized earlier in the function
- Negligible overhead vs the existing O(n²) suffix scan

## Test Scenarios

All tests use synthetic messages. No real session data.

### T1: Fork with no tail overlap → skip
- Session has 50 stored messages (system + 49 user/assistant turns)
- Incoming: 10 messages, none matching stored tail
- Expected: cursor = len(incoming) (skip), no new rows persisted

### T2: Fork with partial head overlap but no tail overlap → skip
- Session has 50 stored messages
- Incoming: 10 messages sharing first 5 with stored head, but last 5 differ
- Expected: skip (tail overlap is the discriminator, not head)

### T3: Legitimate restart with full replay → normal cursor advance
- Session has 20 stored messages
- Incoming: 25 messages (20 replay + 5 new)
- Expected: cursor = 20, only 5 new messages persisted

### T4: Legitimate restart with short delta → normal persist
- Session has 50 stored messages
- Incoming: 5 messages, all matching stored tail suffix
- Expected: cursor advanced past matched suffix, delta persisted

### T5: Legitimate restart, incoming == session_count → guard skipped
- Session has 10 stored messages
- Incoming: 10 messages, no tail overlap (edge case)
- Expected: guard NOT triggered (len < session_count is false), falls through to existing logic

### T6: Very short session + restart → guard skipped
- Session has 3 stored messages
- Incoming: 3 messages, no overlap
- Expected: guard NOT triggered (3 < 3 is false), existing logic handles

### T7: Empty session → early return (existing behavior)
- Session has 0 stored messages
- Incoming: 5 messages
- Expected: cursor = 0 (existing early return), guard never reached

### T8: Fork with tail overlap → guard skipped (false negative)
- Session has 50 stored messages
- Incoming: 10 messages, last 2 match stored tail
- Expected: guard NOT triggered, falls through to existing logic
- Note: this is an accepted false negative — if the fork carries
  parent's tail, re-ingest is less harmful

### T9: Concurrent fork + parent interleaving
- Session has 30 stored messages
- Fork ingests 8 messages (no tail overlap) → skipped
- Parent ingests 35 messages (30 replay + 5 new) → cursor=30, 5 new
- Expected: no duplication, parent's ingest unaffected by skipped fork

### T10: Ignore-pattern-filtered messages
- Session has 40 stored messages, 5 match ignore patterns
- Incoming: 8 messages after filtering, no tail overlap with filtered stored tail
- Expected: skip (filtering is consistent between incoming and stored)
