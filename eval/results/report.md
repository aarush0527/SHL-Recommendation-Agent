# SHL Recommendation Agent -- Quantitative Evaluation Report

Generated: **2026-07-03 01:15:13**

Scores the live system against SHL's own reference conversations (`eval/traces/`, n=10). Companion to `run_tests2.py`'s conversational-behavior suite -- see `eval/README.md` for full methodology, metric definitions, and known limitations of a 10-conversation reference set.

---

## A. Retrieval Quality (offline retriever, isolated from the LLM)

| Metric | Value |
|---|---|
| precision@5 | 0.360 |
| recall@5 | 0.450 |
| f1@5 | 0.382 |
| ndcg@5 | 0.518 |
| ap@5 | 0.381 |
| precision@10 | 0.220 |
| recall@10 | 0.535 |
| f1@10 | 0.302 |
| ndcg@10 | 0.551 |
| ap@10 | 0.399 |
| mrr | 0.867 |

Averaged over 10 reference conversations.

## B. End-to-End Recommendation Quality (live /chat, full conversation replay)

| Metric | Value |
|---|---|
| precision@5 | 0.353 |
| recall@5 | 0.320 |
| f1@5 | 0.320 |
| ndcg@5 | 0.397 |
| ap@5 | 0.273 |
| precision@10 | 0.353 |
| recall@10 | 0.320 |
| f1@10 | 0.320 |
| ndcg@10 | 0.382 |
| ap@10 | 0.255 |
| mrr | 0.733 |
| turn_efficiency | 1.000 |

Averaged over 10 replayed conversations.

### Per-conversation detail

| ID | P@10 | R@10 | F1@10 | MRR | Turns (used/ref) | Final EOC | Error |
|---|---|---|---|---|---|---|---|
| 1 | 0.200 | 0.333 | 0.250 | 0.333 | 4/4 | False |  |
| 2 | 0.667 | 0.400 | 0.500 | 1.000 | 3/3 | True |  |
| 3 | 0.000 | 0.000 | 0.000 | 0.000 | 5/5 | False |  |
| 4 | 0.400 | 0.400 | 0.400 | 1.000 | 3/3 | False |  |
| 5 | 0.667 | 0.400 | 0.500 | 1.000 | 3/3 | False |  |
| 6 | 0.000 | 0.000 | 0.000 | 0.000 | 3/3 | False |  |
| 7 | 0.200 | 0.200 | 0.200 | 1.000 | 4/4 | False |  |
| 8 | 0.400 | 0.400 | 0.400 | 1.000 | 3/3 | False |  |
| 9 | 0.800 | 0.571 | 0.667 | 1.000 | 7/7 | False |  |
| 10 | 0.200 | 0.500 | 0.286 | 1.000 | 3/3 | False |  |

## C. Deterministic Groundedness (whitelist re-check against the live catalog)

- Recommendations checked: **137**
- Violations (hallucinated / mismatched): **0**
- Hallucination rate: **0.0000**

## D. LLM-Judged Groundedness / Faithfulness (free-text reply vs. catalog facts)
> **Limitation:** The LLM judge uses a conservative fact-checking prompt. Replies that contain only conversational framing (e.g., acknowledgements or transitions) and no factual claims about assessment properties may be scored lower than expected because the judge currently treats the absence of verifiable claims as unsupported rather than "not applicable." Deterministic catalog validation (Section C) should therefore be considered the primary groundedness guarantee.

- Conversations judged: **8** (skipped: 2)
- Mean groundedness score: **0.125**
- Fraction fully grounded: **0.125**
- Total unsupported claims flagged: **7**

**Unsupported claims flagged:**

- Conversation 1: "The assistant's reply contains no specific, checkable factual claims about the assessments' properties." -- The reply does not provide any specific information about the assessments that can be verified against the catalog facts.
- Conversation 4: "The assistant's reply does not provide any specific, checkable factual claims about the assessments." -- The reply does not contain any specific information about the assessments that can be verified against the catalog facts.
- Conversation 5: "The OPQ32r takes about 25 minutes to complete" -- The catalog facts support this claim, but the assistant's reply does not make this claim. However, the claim about the duration of OPQ Team Types & Leadership Styles Profile and OPQ MQ Sales Report is not mentioned in the reply, but the catalog facts do not specify the duration for these assessments.
- Conversation 7: "The assistant's reply contains no specific, checkable factual claims about the assessments." -- The reply does not provide any specific information about the assessments that can be verified against the catalog facts.
- Conversation 8: "The assistant's reply contains no specific, checkable factual claims about the assessments' properties." -- The reply does not provide any specific information about the assessments that can be verified against the catalog facts.
- Conversation 9: "Here's what I found based on what you've told me so far." -- This claim is not checkable as it does not provide any specific information about the assessments.
- Conversation 10: "Here's what I found based on what you've told me so far." -- This claim is not specific or checkable against the provided catalog facts.

## E. LLM-Judged Recommendation Relevance (independent of the reference list)
> **Limitation:** Recommendation relevance is inherently subjective. The LLM judge evaluates whether the returned assessments reasonably satisfy the hiring requirements rather than whether they exactly match SHL's reference shortlist. Alternative but well-justified assessment combinations may therefore receive different scores despite being valid recommendations.

- Conversations judged: **8** (skipped: 2)
- Mean relevance precision (relevant=1.0, partial=0.5, not=0.0): **0.650**
- Mean overall relevance score (judge's 1-5 holistic rating): **3.750**

| ID | Relevance precision | Overall score (1-5) | Reason |
|---|---|---|---|
| 1 | 0.600 | 4 | The list includes highly relevant assessments like the Enterprise Leadership Report 2.0 and the OPQ Leadership Report, but also includes some partially relevant or not relevant assessments that do not perfectly fit the stated need. |
| 2 | 0.667 | 4 | The list includes a relevant assessment for networking knowledge, but the cognitive and numerical ability tests, while partially relevant, do not directly assess the specific technical skills required for the role. |
| 4 | 0.800 | 4 | The list includes highly relevant assessments for numerical reasoning and situational judgement, but also includes reports that, while related, do not directly assess candidate abilities. |
| 5 | 0.833 | 4 | The recommended assessments are largely relevant to the stated need of re-skilling the Sales organization, with a good balance of general and sales-specific evaluations. |
| 7 | 0.400 | 2 | The list includes some assessments that are not directly relevant to the stated needs of bilingual healthcare admin staff, particularly the lack of a Spanish-language HIPAA compliance assessment. |
| 8 | 0.600 | 4 | The list includes relevant simulations for both Excel and Word, but also includes some assessments that are only partially relevant or not relevant to the stated need. |
| 9 | 0.800 | 4 | The recommended assessments cover most of the required technical skills, but the inclusion of the Core Java entry-level test is unnecessary and redundant. |
| 10 | 0.500 | 4 | The list includes relevant cognitive and situational judgement assessments for recent graduates, but includes redundant reports and lacks a personality assessment, which was part of the initial request. |

## Latency (end-to-end replay, wall-clock per /chat call)

- Mean: **18.812s** | Median: **20.512s** | Max: **23.795s** | n=38
- Assignment's stated hard limit is 30s/call -- see app/llm.py's budget constants.

## Overall Accuracy / Effectiveness Summary

### Composite effectiveness score: **0.422** / 1.0

Components used (see run_eval.py's `build_overall_summary` docstring for weights and rationale):

| Component | Value |
|---|---|
| end_to_end_f1 | 0.320 |
| llm_relevance_precision | 0.650 |
| llm_groundedness_score | 0.125 |
| turn_efficiency | 1.000 |

Overall observations

• No hallucinated assessments detected.
• Mean latency remained below the assignment's 30-second limit.
• Retrieval consistently placed a relevant assessment near the top (MRR = 0.867).
• Recommendation quality remains the main opportunity for improvement, especially in multilingual and highly specialized scenarios.