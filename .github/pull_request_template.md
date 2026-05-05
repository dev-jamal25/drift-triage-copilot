## Summary

<!-- What does this PR do and why? -->

## Changes

- <!-- list of key changes -->

## Testing

<!-- How was this tested? Any new tests added? -->

## Screenshots

<!-- if UI changes -->

## Checklist

- [ ] Code follows project style guidelines (ruff clean)
- [ ] Self-reviewed the code
- [ ] Added/updated tests
- [ ] Updated documentation (README / ARCH / DECISIONS / RUNBOOK)
- [ ] No secrets or credentials in code
- [ ] No `print()` statements
- [ ] Async correctness verified (no blocking I/O in request paths)
- [ ] Pydantic schemas at every external boundary
- [ ] External calls have timeout + retry
- [ ] If `shared/contracts.py` changed: BOTH services updated and tested

## Linked issues

Closes #<!-- issue number -->
