# Paimon GitHub Master Daily Report

This directory stores the daily analysis state and reports for tracking Apache Paimon GitHub `master`.

## Scope

- Automation repository: `huwh/paimon`
- Upstream source remote: `upstream`, repository `apache/paimon`
- Source branch: `upstream/master`
- Baseline branch: `feature/paimon-1.5-tob-base-phase1-spec`
- Baseline commit: `cc9bb8e165be69c05e615e008d1b9e1a4d26b187`
- Release impact target for bugfixes: `upstream/release-1.4`
- Report timezone: `Asia/Shanghai`

## Daily Process

1. Switch to `codex/paimon-master-daily-report`.
2. Ensure the Apache upstream remote exists:
   `git remote add upstream https://github.com/apache/paimon.git` if it is missing.
3. Fetch `upstream master` and `upstream release-1.4`.
4. Read `state.json`.
5. Analyze commits in first-parent order from `last_processed_commit..upstream/master`.
6. For every commit, inspect the actual patch with `git show --stat --name-status --patch`; do not summarize from the commit message alone.
7. For bugfixes, check whether an equivalent patch is already in `upstream/release-1.4` with `git cherry`, check whether touched production files exist in `upstream/release-1.4`, and then assess the likely impact.
8. Create a daily Markdown report named `YYYY-MM-DD.md`.
9. Update `state.json` only after the report covers all commits through the new `upstream/master` head.

The daily report should start with a short decision-oriented summary, followed by tables for module summary, bugfix impact on `release-1.4`, and commit-level details.
