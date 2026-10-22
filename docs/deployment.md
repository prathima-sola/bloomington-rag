# Cloud Run restoration notes

As observed on October 21, 2026, both the README URL and Cloud Run's reported service URL returned HTTP 503. Read-only inspection confirmed that billing is disabled for the `bloomington-rag-app` project. Cloud Run cannot be restored without enabling billing, so no deployment or billing change was made.

The stored service also has `minScale: 1`. Enabling billing with that setting could create continuous instance usage. A zero-cost-oriented redeployment would first need `minScale: 0`, explicit budget alerts/limits where supported, and user approval for any billing-account change.

Separately, the prior application blocked FastAPI startup while downloading and loading the embedding and reranker models. The server now opens its port immediately and loads retrieval in a background thread, removing that code-level startup failure for a future deployment.

Endpoints:

- `/health` always reports process and loader state.
- `/ready` returns 200 only when retrieval is ready and 503 while loading or after an error.
- `/ask` returns 503 until retrieval is ready.

No deployment or billing configuration is performed by the repository changes. Before redeploying:

1. Revoke and replace the previously configured Groq key; it was exposed during read-only configuration inspection. Never commit the replacement.
2. Decide whether to enable Cloud Billing despite the zero-dollar requirement. Do not proceed without explicit approval.
3. Set minimum instances to zero before enabling billing or deploying.
4. Confirm whether the replacement Groq account/key is strictly free-tier. The deployed linear API currently needs `GROQ_API_KEY`; do not add a paid key without explicit approval.
5. Deploy the revision, then inspect `/health` for model-download, memory, or dependency failures.
6. Configure a readiness probe against `/ready`, not `/health`.
7. Verify one `/ask` request and review Cloud Run usage before leaving the service enabled.

Local Ollama is used by the evaluation harness and agentic pipeline. It is not bundled into the Cloud Run service.
