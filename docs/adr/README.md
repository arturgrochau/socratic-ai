# Architecture Decision Records

Each ADR captures one significant decision: its **context**, the **decision**, its **consequences**
(including honest known trade-offs), and the **alternatives** considered. They are append-only history
— supersede an old ADR with a new one rather than rewriting it.

Format: lightweight [MADR](https://adr.github.io/madr/). Status is one of `accepted`, `superseded`.

| # | Title | Status |
|---|---|---|
| [0001](0001-single-process-nicegui-fastapi.md) | Single process: NiceGUI mounted on FastAPI | accepted |
| [0002](0002-iterative-accumulation-ledger.md) | Iterative-accumulation knowledge ledger | accepted |
| [0003](0003-dual-model-aux-routing.md) | Dual-model (generation + aux) routing | accepted |
| [0004](0004-cache-via-schema-version.md) | Caching via schema-version constants | accepted |
| [0005](0005-cross-source-grounding.md) | Cross-source intersection grounding | accepted |
| [0006](0006-pluggable-llm-providers.md) | Pluggable LLM providers (OpenAI/Ollama) | accepted |
| [0007](0007-single-source-enrichment.md) | Single-source enrichment (quiz/apply-it for one doc) | accepted |
| [0008](0008-map-reduce-ledger.md) | Map-reduce ledger extraction (parallel windows) | accepted |
| [0009](0009-code-validator-and-heuristic-tier.md) | Code validator + heuristic chat tier + focused quiz call | accepted |
