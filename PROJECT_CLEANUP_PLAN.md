# Project Cleanup Plan

## Scope
This plan focuses on safe cleanup that reduces redundancy and inefficiency without breaking the running app.

## Current Structure Audit

### Runtime artifacts and local state
- Present and correctly ignored by .gitignore:
  - uploads/
  - chroma_data/
  - *.db and *.sqlite3
  - .venv/
  - __pycache__/
- Recommendation:
  - Keep these ignored.
  - Add a lightweight maintenance script later for optional local cleanup (not automatic).

### Wrapper scripts
- Files observed:
  - socratic-ai (bash launcher)
  - socratic_ai (bash forwarder to socratic-ai)
- Assessment:
  - Both are useful for compatibility, but this is duplicated entrypoint surface.
- Recommendation:
  - Keep one canonical launcher: socratic-ai.
  - Keep socratic_ai temporarily as a compatibility shim.
  - Mark socratic_ai as deprecated in README and remove in a later release.

### Backend generation pipeline overlap
- Current overlap sources:
  - Summary -> deep dive -> under surface can still repeat concepts if model output is verbose.
  - Multiple prompt stages can over-elaborate the same mechanism.
- Current mitigation already in place:
  - Deterministic dedup filter.
  - Progression outline grounding.
- Recommendation:
  - Keep one dedup implementation in app/generation.py as the only post-pass.
  - Avoid adding stage-specific custom dedup logic in multiple modules.

### Prompt-layer overlap
- Prompt files are separated by concern and should remain separated.
- Redundancy risk is in instruction drift across files.
- Recommendation:
  - Add one short shared style checklist section template in future prompt maintenance.
  - Keep each prompt's domain-specific rules local.

## Redundant or inefficient patterns to consolidate
1. Launcher duplication
- Consolidate to one canonical script while preserving temporary compatibility alias.

2. Similarity logic sprawl risk
- Keep all sentence similarity and overlap thresholds centralized in app/generation.py constants.
- Avoid copying thresholds into tests or prompt files.

3. Formatting logic
- Keep prose formatting helper centralized in frontend/app.py and reuse for deep dive, under-surface, and reflection.

## Staged Cleanup Execution Plan

### Stage 1 (low risk, immediate)
- Keep current ignores and artifacts policy.
- Keep one canonical launcher documented (socratic-ai).
- Ensure README references one launcher path.

### Stage 2 (low-medium risk)
- Add deprecation note for socratic_ai shim.
- Add a release note that socratic_ai will be removed after one stable cycle.

### Stage 3 (medium risk)
- Remove socratic_ai shim once docs and users have migrated.
- Verify no scripts, CI jobs, or deployment hooks call socratic_ai.

### Stage 4 (quality hardening)
- Periodically tune dedup and efficiency thresholds using generation_stage_events telemetry.
- Keep short-source heuristic and model-path decision in one helper to prevent future branching duplication.

## Validation Checklist After Cleanup Changes
1. Launch app using canonical script.
2. Run tests:
   - tests/test_generation_quality.py
   - tests/test_optional_source_modes.py
3. Run manual flow for:
   - video-only
   - docs-only
   - mixed
4. Verify no references to removed/deprecated launcher remain in docs or scripts.
