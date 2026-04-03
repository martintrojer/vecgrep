# Product Philosophy

vecgrep is semantic `ripgrep` for a repository.

It exists to make local, path-scoped search better without changing the core mental model that makes grep useful: one root, explicit paths, predictable output, fast feedback, and easy composition with other tools. The goal is not to become a general retrieval platform, knowledge base, or agent backend. The goal is to help users find the right code or notes in a repo, especially when exact keywords are not enough.

The product should stay narrow on purpose. Its strength comes from being local-first, scriptable, and understandable. Features are good only when they reinforce that shape. Hybrid search fits when it acts as an explicit ranking mode for grep-like queries. It does not fit when it introduces hidden state, overloaded flags, or platform-style abstractions that weaken predictability.

Predictability is a core product value, not just an implementation concern. Query mode, index capability, path scope, and output state should be explicit to the user and clearly separated in the code. If a feature makes vecgrep harder to reason about, harder to compose, or harder to explain, it is probably pushing the tool in the wrong direction.

Success for vecgrep is not “more AI.” Success is being the fastest, clearest, most trustworthy way to do semantic search in a repo while still feeling like a tool you can script, debug, and replace with `rg` when exact matching is the right answer.

## Council Addendum

This philosophy was reinforced by a multi-perspective council discussion covering architecture, UX, engineering, and product positioning. The strongest point of agreement was that vecgrep is most valuable when it remains a narrow tool with a stable mental model, not when it expands into a broader retrieval system.

The council also converged on a more specific claim: predictability is not just a technical implementation goal, it is part of the product itself. Users should be able to understand what root is selected, what paths are in scope, which search mode is active, and whether the tool is behaving differently because of index state or explicit flags. When those distinctions blur, trust drops quickly.

### Why This Direction Is Strong

- vecgrep has a clear identity: semantic `ripgrep` for a repo is easier to explain, use, and defend than a vague local AI search platform.
- The single-root, local-first model keeps behavior understandable and operationally light.
- Path scoping and ripgrep-like conventions make the tool feel familiar instead of magical.
- Unix composition remains a major advantage: users can still combine `vecgrep` with `rg`, `jq`, `xargs`, editors, and shell pipelines.
- Opt-in hybrid search fits the philosophy when it improves grep-like queries without changing the default story.
- A narrow product is easier to benchmark honestly and easier to maintain without hidden policy layers.

### Risks And Failure Modes

- Product drift is the main danger. Features can slowly turn vecgrep into a retrieval platform without any single obviously bad decision.
- Overloaded semantics are especially dangerous. A single flag or config field should not silently affect query behavior, index capability, and output meaning at the same time.
- Invisible modes create UX debt. Even a small tool feels confusing if users cannot tell when hybrid or other non-default behavior is active.
- Benchmark variance across corpora and embedders is a warning against making one retrieval policy the universal default.
- Extra abstractions such as collections, document IDs, or additional execution stacks would increase maintenance cost and weaken the core repo-native mental model.

### Practical Implications

- Default behavior should stay boring, fast, and easy to reason about.
- Advanced behavior should remain explicit, visible, and honestly documented.
- Query mode, index capability, and user-facing state should stay distinct concepts in both code and docs.
- New features should be judged against a simple question:
  does this make vecgrep a better semantic grep, or does it push it toward becoming something else?

### Pros And Cons Of The Current Shape

**Pros**

- Clear repo-native model
- Strong local-first story
- Good fit for terminal, editor, and scripting workflows
- Easy to describe in one sentence
- Hybrid can add value without redefining the tool

**Cons**

- The narrow scope limits adjacent feature expansion
- Some powerful ideas from broader retrieval systems do not fit cleanly
- Hybrid and other advanced modes require careful visibility to avoid confusion
- The product must resist tempting abstractions even when they look useful in isolation
- Maintaining conceptual clarity gets harder as capabilities accumulate
