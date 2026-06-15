# Paimon GitHub Master Daily Report

This directory stores the daily analysis state and reports for tracking Apache Paimon GitHub `master`.

## Scope

- Source remote: `github`, repository `apache/paimon`
- Source branch: `github/master`
- Baseline branch: `feature/paimon-1.5-tob-base-phase1-spec`
- Baseline commit: `cc9bb8e165be69c05e615e008d1b9e1a4d26b187`
- Release impact target for bugfixes: `github/release-1.4`
- Report timezone: `Asia/Shanghai`

## Daily Process

1. Switch to `codex/paimon-master-daily-report`.
2. Fetch `github master` and `github release-1.4`.
3. Read `state.json`.
4. Analyze commits in first-parent order from `last_processed_commit..github/master`.
5. For every commit, inspect the actual patch with `git show --stat --name-status --patch`; do not summarize from the commit message alone.
6. For bugfixes, check whether an equivalent patch is already in `github/release-1.4` with `git cherry`, check whether touched production files exist in `github/release-1.4`, and then assess the likely impact.
7. Create a daily Markdown report named `YYYY-MM-DD.md`.
8. Update `state.json` only after the report covers all commits through the new `github/master` head.

The daily report should start with a short decision-oriented summary, followed by tables for module summary, bugfix impact on `release-1.4`, and commit-level details.
