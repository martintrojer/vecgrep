# Product Philosophy

vecgrep is semantic `ripgrep` for a repository.

It exists to make local, path-scoped search better without changing the core mental model that makes grep useful: one root, explicit paths, predictable output, fast feedback, and easy composition with other tools. The goal is not to become a general retrieval platform, knowledge base, or agent backend. The goal is to help users find the right code or notes in a repo, especially when exact keywords are not enough.

The product should stay narrow on purpose. Its strength comes from being local-first, scriptable, and understandable. Features are good only when they reinforce that shape. Hybrid search fits when it acts as an explicit ranking mode for grep-like queries. It does not fit when it introduces hidden state, overloaded flags, or platform-style abstractions that weaken predictability.

Predictability is a core product value, not just an implementation concern. Query mode, index capability, path scope, and output state should be explicit to the user and clearly separated in the code. If a feature makes vecgrep harder to reason about, harder to compose, or harder to explain, it is probably pushing the tool in the wrong direction.

Success for vecgrep is not “more AI.” Success is being the fastest, clearest, most trustworthy way to do semantic search in a repo while still feeling like a tool you can script, debug, and replace with `rg` when exact matching is the right answer.
