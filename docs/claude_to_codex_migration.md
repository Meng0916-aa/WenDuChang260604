# Claude To Codex Migration Report

This report records the repository-local Claude artifact audit and the migration
status for Codex takeover. It is a migration record, not a formal configuration
source.

## Summary

- Formal rule entry: `AGENTS.md`.
- Multi-file planning standard: `PLANS.md`.
- Current project state: `docs/project_state.md`.
- Formal decisions: `docs/decisions.md`.
- Current next task: `docs/next_task.md`.
- `.claude/` remains in place as historical/local Claude configuration.
- `.agents/` was removed because it contained only empty temporary skill drafts.

## Source Classification

| Claude source file | Original purpose | Classification | Migrated | Migration target | Rewritten content | Not migrated reason | Validation status |
|---|---|---|---|---|---|---|---|
| `.claude/skills/experiment-reproducibility/SKILL.md` | Intended project skill | E. obsolete/empty | No | None | None | File is 0 bytes; no project content to migrate. | Checked length = 0 |
| `.claude/skills/paper-result-analysis/SKILL.md` | Intended project skill | E. obsolete/empty | No | None | None | File is 0 bytes; no project content to migrate. | Checked length = 0 |
| `.claude/skills/thermal-cycle-prediction/SKILL.md` | Intended project skill | E. obsolete/empty | No | None | None | File is 0 bytes; no project content to migrate. | Checked length = 0 |
| `.claude/skills/xtherm-data-pipeline/SKILL.md` | Intended project skill | E. obsolete/empty | No | None | None | File is 0 bytes; no project content to migrate. | Checked length = 0 |
| `.claude/settings.local.json` | Claude Code local permissions | D. Claude-specific content | No | None | Project intent is already covered by `AGENTS.md` and `PLANS.md`; broad tool permissions were not copied. | Claude-local permission syntax is not authoritative for Codex and includes commands that conflict with current rules. | Structure checked; no sensitive keyword pattern found |

## Content Migrated Into AGENTS.md

- Current formal pipeline status.
- Authoritative formal configuration list.
- Key confirmed formal parameters.
- Hard constraints for raw data, generated data, Git, dependency installation,
  legacy configuration, protected local files, and conflict handling.
- Testing environment rule for the full `pytorch` test suite.
- Rule that `.claude/` is not the current authoritative entry.
- Rule that task-relevant `docs/workflows/` documents must be read when they
  exist.

## Content Migrated Into PLANS.md

- Multi-file task plans must be file-scoped.
- Plans must define purpose, scope, inputs, allowed and forbidden files,
  milestones, validation, rollback, open questions, and completion criteria.
- Plans must not silently expand scope.
- Configuration, code, and documentation conflicts require stopping.
- Local `results/` history is not evidence of formal pipeline completion.

## Content Migrated Into docs/workflows/

No workflow documents were created. The repository-local Claude skill files were
empty, so there was no reusable workflow content to migrate. Do not create a
Codex-specific hidden skill directory unless the client-supported location is
confirmed in a future task.

## Content Still Retained In .claude/

- `.claude/settings.local.json`
- `.claude/skills/experiment-reproducibility/SKILL.md`
- `.claude/skills/paper-result-analysis/SKILL.md`
- `.claude/skills/thermal-cycle-prediction/SKILL.md`
- `.claude/skills/xtherm-data-pipeline/SKILL.md`

These files are retained because this task did not authorize deleting
`.claude/`.

## Expired Or Not Migrated Content

| Source | Expired or not migrated content | Reason not migrated | Replacement authority |
|---|---|---|---|
| `.claude/settings.local.json` | Claude Code permission allow-list, including broad Git, pip, shell, and GitHub CLI commands | Claude-specific permission syntax; several allowances conflict with current hard constraints | `AGENTS.md`, `PLANS.md` |
| `.claude/skills/*/SKILL.md` | Empty skill placeholders | No content | `AGENTS.md`, `PLANS.md`, `docs/project_state.md`, `docs/decisions.md`, `docs/next_task.md` |
| `.agents/skills/*/SKILL.md` | Empty temporary skill placeholders | Temporary/empty duplicate of skill names; removed after audit | No replacement needed |

## Claude-Specific Dependencies

Claude-specific local artifacts are present only as `.claude/settings.local.json`
and empty skill placeholders. No Claude commands, hooks, agents, or reusable
workflow bodies were found in repository-local `.claude/`.

## Sensitive Information

No token, secret, password, API-key, bearer-token, credential, or private-key
keyword pattern was found in `.claude/settings.local.json` during the migration
audit. No secret values are reproduced in this report.

## Notes

- `scripts/02d_conversion_report.py` is a protected local untracked file and is
  not a version-controlled formal pipeline component.
- The protected Word document `项目进展与下一步数据提取说明.docx` was not read or
  modified.
- Formal ROI generation and formal feature extraction remain closed.
- `docs/real_data_import.md` was reconciled so that the formal direct `.xtherm`
  parser path and the legacy exported-matrix workflow are clearly separated.
