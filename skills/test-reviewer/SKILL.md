---
name: test-reviewer
description: Use this skill to review tests for false confidence, excessive mocking, meaningless assertions, and weak behavior coverage. Use after writing tests or when a suite passes but bugs still escape.
version: 0.1.0
---

# Test Reviewer

## Purpose

Evaluate whether tests provide real safety signals or merely appear to do so.

## What to Look For

### 1. Excessive Mocking
- Mocks outnumber real objects
- Mock setup mirrors production logic
- Tests pass regardless of implementation correctness
- Mock verification replaces behavior verification

### 2. Fake Testing
- Assertions recreate implementation instead of validating outcomes
- Tests verify that mocks were called, not user-visible behavior
- Tests compare a value to itself or otherwise tautological checks
- Tests that execute code with no meaningful assertion

### 3. Weak Assertions
- Always-true or trivial assertions
- Missing assertions
- Hardcoded expected values that do not constrain behavior
- Assertions that would still pass for broken logic

### 4. Test Smells and Coverage Quality
- Excessive irrelevant setup
- Brittle tests tightly coupled to implementation details
- Order-dependent or global-state-dependent tests
- Critical paths and edge cases untested

## Review Process

1. Check if tests fail when behavior is broken
2. Evaluate whether mocks are necessary and justified
3. Confirm test independence and deterministic setup
4. Identify gaps in behavior coverage and edge cases
5. Recommend stronger, behavior-focused rewrites

## Output Format

Use this structure:

### Critical Issues
Tests that provide false confidence.

### Recommended Changes
High-impact rewrites that improve confidence.

### Suggestions
Minor improvements for clarity and maintainability.

For each finding:
- Quote the problematic test code
- Explain what bug could slip through
- Provide a concrete improved test
- State what behavior the improved test verifies

## Principles

- A test must fail when behavior is wrong
- Prefer behavior assertions over implementation assertions
- Use real dependencies when practical
- One test should validate one behavior
- Passing tests should correlate with working software
