# TensorRT-RTX permanent worktree migration

## Scope and verification contract

Move the existing NVIDIA/Gemma 4 12B work onto freshly fetched Google AI Edge
main branches, preserve upstream changes, verify actual execution and memory,
then retire the named old directories without losing source history or results.
Review NVIDIA-specific code for demonstrably redundant work; do not trade
correctness, cache integrity, or throughput for a shorter implementation.

The sibling names follow `gitwt trt_rtx` (`<repository>_trt_rtx`). Native
`git worktree add` supplies `origin/main` explicitly because `gitwt` starts at
the current HEAD rather than accepting a base revision.

Fetched upstream baselines (2026-09-05):

- LiteRT: `761d99cb90e20c67efcb3fe1119a60c92381bd1a`.
- LiteRT-LM: `007e7c760efc3dae5c1d628ee8f28053ae149468`.

Permanent checkouts and tracking branches:

- `/home/lijin/odml/rt_g3/LiteRT_trt_rtx`, branch `LiteRT_trt_rtx`.
- `/home/lijin/odml/llm/LiteRT-LM_trt_rtx`, branch `LiteRT-LM_trt_rtx`.
- Both branches track the corresponding Google AI Edge `origin/main`.

## September 6 history cleanup

The user-approved cleanup retains only local commits whose original author
dates are September 4, 2026 or later: 15 LiteRT commits (11 source-work
milestones and four migration/reporting milestones) and two LiteRT-LM commits.
All retained commits use `litert-g-developer <litert-g-developer@gg.com>` as
author and committer; original messages, author dates, order, and individual
commit boundaries are preserved. Upstream history is not re-authored.

The branches now build on the Google AI Edge main revisions fetched September 6:

- LiteRT: `cc1850a02ed3a351e7ea5076e085cf6954e4ef07`.
- LiteRT-LM: `a345f8ba60141e731cef183f43d99b82645e8e90`.

The 16 earlier local LiteRT commits are omitted from this series, including
13 empty preservation commits and the two intentionally excluded README
commits. The remaining earlier commit's kernel copyright/license attribution
is folded into the retained documentation-and-attribution milestone. Later
README edits are omitted as well; runtime code and run-script behavior are
not removed along with those commits.

Both repositories retain `codex/backup/trt-rtx-before-cleanup-20260906`, pointing
to the pre-cleanup tips `53d384a3b` (LiteRT) and `6a31deac` (LiteRT-LM).
`migration_commits.tsv` maps the 13 retained original source-work commits to
their current hashes. The four later migration/reporting commits remain
separate milestones; they are not entries in that source-commit manifest.

The September 5 measurements below and in `trt_rtx_measurements.md` remain
historical evidence for the September 5 bases, not a fresh performance
measurement of the September 6 upstream changes.

## September 5 migration milestones (historical)

- [x] Inspect remotes, source state, worktree naming, cleanup targets, and disk.
- [x] Fetch both Google AI Edge main branches and create sibling worktrees.
- [x] Preserve all 27 local LiteRT commits (original NVIDIA work, PR follow-ups,
      and 12B work) and both LiteRT-LM commits individually. See
      the September 5 backup's `migration_commits.tsv` for that historical map.
- [x] Verify original messages, author names, author emails, and author dates
      match all 29 migrated commits. Original branches remain recoverable.
- [x] Preserve remaining local documentation and licensing differences.
- [x] Build both NVIDIA libraries and both LiteRT-LM executables here.
- [x] Run compiler, dispatch, graph-pruning, buffer, and signature tests
      (11 LiteRT targets and 2 LiteRT-LM targets passed).
- [x] Pass protected benchmark preflight; check CPU/NPU generation agreement
      (the 12B CPU reference, cold AOT, and warm AOT all returned `Paris`).
- [x] Verify fresh AOT creation, AOT reuse, and fresh in-memory JIT for 12B;
      verify the E2B all-signature path also returns `PARIS` on NPU and CPU.
- [x] Repeat uninstrumented decode/prefill measurements and granular memory passes.
      Same-day paired throughput and warm memory agree. Cold compilation was
      slower in the new runs and remains an explicitly recorded open finding;
      see `trt_rtx_measurements.md` for the limits of the performance conclusion.
- [x] Archive unique history/results, remove only approved old locations, and
      update the two original checkouts to upstream main. Original refs and
      detached/stale worktree revisions remain recoverable.
- [x] Review NVIDIA integration in both repositories and document retained
      safety boundaries in `trt_rtx_review.md`. No runtime simplifications are
      mixed into the migration; proposed changes need separate measurements.
- [x] Pass a post-cleanup CPU/NPU output check and AOT hit without recompilation.
- [x] Commit results and confirm the final worktree/dependency inventory.

## Existing comparison reference

The September 5 migration retained already-upstream commits as documented
empty commits, rather than reverting upstream fixes to make old patches apply.
That history reconstruction used an isolated Git index and verified that its final
tree stayed byte-for-byte identical (`b414e59477e6e196ea10f9272d96e603f8a4b042`)
to the initial migrated tree; ongoing builds did not see source files change.

The historical results in `gemma4_12b.md` are not measurements of this new
upstream base. Acceptance requires fresh measurements. Keep the 12B checkpoint,
2,048-token context, 1,024-token prefill, 256-token decode, selected
`prefill_1024,decode` signatures, BF16 activations, `cuda_gemv`, shared weights,
and 512 MiB FC weight cap fixed. Use separate profiling and throughput runs.

Do not interpret passing builds as an operational result. Preserve failed
runs, report swap separately from RSS, distinguish runtime-cache cold/warm,
and do not remove a validation boundary merely because it is NVIDIA-specific.

## Final locations

The two permanent branches contain the retained September 4-and-later commits
individually, not a squashed source copy. Use `git log --oneline origin/main..HEAD`
in each worktree and `migration_commits.tsv` to map the original source commits.
The September 6 cleanup explicitly changes author identities while preserving
messages and author timestamps. New parents and metadata change Git hashes.

The original LiteRT and LM directories are clean on `main` at the September 5
baselines above. Prior topic branches and main-branch backups remain in their
Git repositories. The four retired directories are archived at
`/home/lijin/odml/archive/trt_rtx_migration_20260905`; the independent old
`litert_cuda` repository additionally has a verified all-refs bundle. Linked
archived worktrees were repaired, and their refs compare exactly before/after.
Only the clean temporary working copy and stale registrations were removed.

Historical results are in `results/previous_runs`, current measurements in
`results/migration_20260905`, and `trt.patch` remains untracked. The two shell
exports in `/home/lijin/dotfiles/shells/odml_litert.sh` now point to the permanent
worktrees (local dotfiles commit `0e6cf780`); model and SDK defaults are unchanged.
No remote push or PR modification was performed.
