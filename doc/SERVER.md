# Server API

See also: [User Guide](GUIDE.md) | [Configuration](CONFIG.md)

Start vecgrep as an HTTP server for IDE integration and scripting:

```bash
vecgrep --serve --port 8080 ./src
# => Listening on http://127.0.0.1:8080
```

The server loads the embedding model once and keeps the index warm. Each query is a simple HTTP request — no startup overhead.

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /search?q=<query>&k=<N>&threshold=<F>` | Semantic search, returns JSONL |
| `GET /status` | Pipeline status as JSON |

### `/search`

```bash
curl -s "http://localhost:8080/search?q=error+handling&k=5"
```

Returns JSONL (one JSON object per line):

```json
{
  "root": "/path/to/project",
  "file": "src/main.rs",
  "start_line": 42,
  "end_line": 58,
  "score": 0.847,
  "text": "..."
}
```

Parameters:
- `q` (required) — search query
- `k` (optional) — max results, defaults to server's `--top-k`
- `threshold` (optional) — min similarity, defaults to server's `--threshold`

### `/status`

```bash
curl -s "http://localhost:8080/status"
```

Returns JSON:

```json
{
  "status": "indexing",
  "indexed": 42,
  "total": 380,
  "chunks": 85,
  "version": "0.9.1",
  "root": "/path/to/project"
}
```

```json
{
  "status": "ready",
  "files": 380,
  "chunks": 850,
  "version": "0.9.1",
  "root": "/path/to/project",
  "scope": ["src"]
}
```

Fields:
- `status` — `"indexing"` or `"ready"`
- `total` — `null` while the file walker is still scanning
- `version` — vecgrep binary version
- `root` — project root path
- `scope` — active path scopes (omitted when searching the full project)

IDE plugins can poll this to show indexing progress or wait for readiness.

## Usage with fzf

```bash
vecgrep --serve --port 8080 ./src &
fzf --bind "change:reload:curl -s 'http://localhost:8080/search?q={q}'" --preview 'echo {}'
```

## Integrations

- [vecgrep.nvim](https://github.com/martintrojer/vecgrep.nvim) — Neovim plugin for semantic search via `--serve` mode
