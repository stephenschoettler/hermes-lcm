# Stage 2 Bot-Pass Finding Verdicts

Source: PR [#173](https://github.com/100yenadmin/hermes-lcm/pull/173) review comments created after `2026-07-29T09:10:00Z`, fetched with the command in `SPEC-STAGE2-BOTPASS.md`.

Sixteen comments matched the cutoff. Seven `bench/*.md` or `bench/specs/*.md` comments were excluded without inspection or edits, as required. All nine code comments reproduced on `eb477f81a089766205136d2ffcb590c61e01ddcb` and have the terminal disposition **fixed now**:

| Comment | Verdict and fix | Regression proof |
|---|---|---|
| [3672704120](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672704120) | Valid: positive out-of-range timestamps escaped the conversion guard. The UTC conversion now fails closed on `ValueError`, `OverflowError`, or `OSError`. | `test_session_expand_v1_date_is_date_only_and_fails_closed_out_of_range` |
| [3672704139](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672704139) | Valid: the flag-OFF bridge forced `answer_ready`. The bridge now adds that mode only when `session_expand_v1` is enabled. | `test_real_bridge_flag_off_uses_historical_recall_mode`; pinned-base byte comparison |
| [3672704156](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672704156) | Valid: post-strict contiguity repeatedly scanned `candidate_records`. Records are indexed once by `(session_id, ordinal)`. | `test_session_expand_v1_routes_stale_window_rows_through_strict_backstop` |
| [3672710101](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710101) | Valid: the post-strict near-failure/far-drop branch lacked coverage. The stale-row regression now asserts the farther row is dropped and the metric is `1`. | `test_session_expand_v1_routes_stale_window_rows_through_strict_backstop` |
| [3672710111](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710111) | Valid: the Bun seam probe could hang. It now has `timeout=120`. | Full `tests/test_stage2_bridge.py` pass with the real harness seam |
| [3672710147](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710147) | Valid: fallback dates mixed datetime and date-only shapes. They now use UTC `date().isoformat()`. | `test_session_expand_v1_date_is_date_only_and_fails_closed_out_of_range` |
| [3672710160](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710160) | Valid: expansion loaded whole sessions. It now pages outward in bounded 12-row anchor windows while retaining a captured tail bound. | `test_session_expand_v1_pages_from_anchor_without_full_session_reads` |
| [3672710166](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710166) | Valid: contiguity issued repeated fresh `COUNT(*)` queries. The ordinal walk now uses the bounded snapshot records and performs no count query. | `test_session_expand_v1_pages_from_anchor_without_full_session_reads` |
| [3672710173](https://github.com/100yenadmin/hermes-lcm/pull/173#discussion_r3672710173) | Valid: the treatment response cap was derived from its payload and therefore uncapped. It now has an operator cap bounded by the historical 64,000 floor and an absolute 512,000 ceiling, with configuration and payload clamp metrics. | `test_session_expand_v1_config_defaults_and_environment`; `test_session_expand_v1_clamps_the_treatment_response_envelope` |

Validation: focused files `103 passed`; Ruff passed; flag-OFF base/candidate files are byte-identical at 13,742 bytes and SHA-256 `0480a8444845ea7226861471fd00cd6d9c0fb5daa9d8600b7d0feab81d25d875`; isolated full suite `2,754 passed, 1 skipped, 12 xfailed`. Evidence is under `/Volumes/LEXAR/Codex/session-notes/2026-07-29/hermes-r3-0/artifacts/laneS2B-logs/`.
