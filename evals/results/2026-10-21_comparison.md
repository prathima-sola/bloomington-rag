| Config | Pipeline | N | Hit@3 | MRR | Recall@5 | Faithfulness | Refusal acc. | p50 retrieval ms | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense_only | linear | 50 | 0.940 | 0.838 | 0.930 | — | — | 4 | 2026-10-21_dense_only.json |
| hybrid_no_rerank | linear | 50 | 0.980 | 0.935 | 0.970 | — | — | 6 | 2026-10-21_hybrid_no_rerank.json |
| baseline_linear | linear | 50 | 0.920 | 0.890 | 0.940 | — | — | 45 | 2026-10-21_baseline_linear.json |
| baseline_langgraph | langgraph | 50 | 0.960 | 0.927 | 0.950 | — | — | 45 | 2026-10-21_baseline_langgraph.json |
