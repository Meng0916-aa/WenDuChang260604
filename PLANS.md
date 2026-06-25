# Multi-File Task Planning Standard

All complex or multi-file tasks must start with a file-scoped plan before any
implementation.

Each plan must contain:

1. Task purpose.
2. Modification scope.
3. Authoritative inputs.
4. Files allowed to modify.
5. Files forbidden to modify.
6. Implementation milestones.
7. Validation commands.
8. Rollback plan.
9. Open questions.
10. Completion criteria.

Planning rules:

- The plan must not silently expand task scope.
- Do not implement across pipeline stages without user approval.
- Stop when configuration, code, and documentation disagree.
- Do not substitute guesses for repository evidence.
- Do not treat historical files in local `results/` as evidence that the formal
  pipeline stage is complete.

For this repository, the current formal state is recorded in
`docs/project_state.md`, formal decisions are recorded in `docs/decisions.md`,
and the next approved task is recorded in `docs/next_task.md`.
