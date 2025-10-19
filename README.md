# Agent Memory
Project on Agent Memory Management

Overview
- A LangGraph-based agent (LangChain tools: DuckDuckGo search + Python REPL) served via FastAPI.
- Two endpoints:
  - `POST /v1/agent/respond` (baseline naive tools)
  - `POST /v1/langgraph/respond` (LangGraph agent)
- No long-term memory in baseline; evaluation focuses on context/memory weaknesses.

Run locally
```
pip install -r requirements.txt
export OPENAI_API_KEY=...  # or set in your shell
uvicorn app.main:app --reload --port 8000
```

Docker
```
docker build -t agent:latest .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY agent:latest
```

Cloud deployment (AWS ECS Fargate)
- See `infra/terraform/README.md` for details. The ALB routes based on `X-User-Id` header to per-user isolated ECS services.

API examples
```
curl -X POST http://localhost:8000/v1/langgraph/respond \
  -H "Content-Type: application/json" \
  -H "X-User-Id: userA" \
  -d '{"message":"search for latest LangGraph release","session_id":"s1"}'
```

Evaluation
- Benchmark file: `eval/benchmark.json` defines multiple users and sessions with mixed context/tool needs.
- Runner: `python eval/run_eval.py` (set `EVAL_BASE_URL` to ALB DNS or `http://localhost:8000`).
  - Outputs CSV lines: User, Session, NumRequests, AvgLatencyMs, SearchHits, PythonHits.

Notes
- This is a baseline with intentionally weak context/memory handling. Use it to test memory system improvements later.
