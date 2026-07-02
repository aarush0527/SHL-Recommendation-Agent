# Conversational SHL Assessment Recommendation Agent

> An intelligent conversational recommendation system that helps recruiters discover the most relevant **SHL assessments** through natural language conversations using deterministic routing, explainable retrieval, and LLM-assisted dialogue.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LLM](https://img.shields.io/badge/Meta%20Llama%203.3%2070B-Groq-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Recruiters rarely know the exact assessment they need. Instead, they describe hiring requirements conversationally:

> *"I'm hiring a senior Java backend engineer with strong stakeholder communication skills."*

Traditional catalog search relies on manually selecting filters or knowing assessment names beforehand.

This project explores a conversational approach by combining deterministic application logic, explainable retrieval, and large language models to transform hiring conversations into grounded recommendations from the **SHL Individual Test Solutions Catalog**.

The agent can:

- Ask clarifying questions when requirements are incomplete
- Recommend relevant assessments once enough context is available
- Refine recommendations as user constraints change
- Compare assessments using catalog evidence
- Refuse requests outside the supported domain

---

## Features

- 💬 Multi-turn conversational recommendations
- ❓ Intelligent clarification of vague hiring requests
- 🔄 Dynamic recommendation refinement
- ⚖️ Grounded assessment comparison
- 🔍 Explainable TF-IDF retrieval
- 📚 Catalog normalization pipeline
- 🛡️ Hallucination-resistant recommendation generation
- 📋 Automatic schema validation
- ⚡ Retry and graceful fallback handling
- 🚀 Stateless FastAPI REST API
- 📊 Automated evaluation framework

---

## System Architecture

```text
                 Conversation History
                         │
                         ▼
        Structured Requirement Extraction
                  (Meta Llama 3.3)
                         │
                         ▼
          Deterministic Conversation Router
                         │
      ┌──────────┬──────────┬──────────┐
      │          │          │          │
   Clarify    Recommend   Compare    Refuse
                         │
                         ▼
               TF-IDF Retrieval Engine
                         │
                         ▼
          Structured Relevance Boosting
                         │
                         ▼
         Grounded Response Generation
                         │
                         ▼
          Catalog Validation Layer
                         │
                         ▼
                 JSON API Response
```

---

## Design Philosophy

Rather than allowing the language model to control the application's behaviour, the system separates **language understanding** from **application control**.

The LLM is responsible for:

- Requirement extraction
- Intent understanding
- Natural language generation

The application controls:

- Conversation routing
- Retrieval
- Candidate ranking
- Recommendation validation
- Schema enforcement
- Catalog grounding

This separation produces a system that is more explainable, reproducible, and easier to debug and evaluate.

---

## Retrieval Pipeline

The recommendation engine is built over the complete **SHL Individual Test Solutions Catalog**.

Retrieval follows four stages:

1. Catalog normalization
2. TF-IDF lexical retrieval
3. Structured relevance boosting
4. Recommendation validation

Instead of hard filtering, structured requirements are incorporated as additive ranking signals. This preserves recall while still prioritizing the strongest matches.

Every recommendation remains fully traceable back to catalog data.

---

## Evaluation

The project includes both quantitative and qualitative evaluation.

### Retrieval Metrics

- Precision
- Recall
- F1 Score
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- nDCG

### Behaviour Testing

The evaluation suite includes scenarios covering:

- Vague hiring requests
- Multi-turn clarification
- Recommendation refinement
- Assessment comparison
- Prompt injection attempts
- Off-topic requests
- Typographical errors
- Contradictory requirements
- Catalog boundary cases

Retrieval quality and end-to-end conversational behaviour are evaluated independently to simplify regression analysis.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Language | Python |
| LLM | Meta Llama 3.3 70B Versatile (Groq) |
| Retrieval | TF-IDF + Cosine Similarity |
| Validation | Pydantic |
| HTTP Client | httpx |
| Deployment | Render |

---

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── router.py
│   ├── llm.py
│   ├── retrieval.py
│   ├── prompts.py
│   ├── guardrails.py
│   └── ...
│
├── data/
│   ├── catalog.json
│   └── ...
│
├── eval/
│   ├── run_eval.py
│   ├── metrics.py
│   └── ...
│
├── tests/
├── requirements.txt
└── README.md
```

---

## Running Locally

```bash
git clone https://github.com/<your-username>/<repository>.git

cd <repository>

pip install -r requirements.txt

uvicorn app.main:app --reload
```

The API exposes:

```http
GET /health
POST /chat
```

---

## Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Hiring a senior Java backend engineer with stakeholder communication skills."
    }
  ]
}
```

---

## Example Response

```json
{
  "reply": "Based on your requirements, here are several relevant assessments.",
  "recommendations": [
    {
      "name": "...",
      "url": "...",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

---

## Engineering Highlights

- Deterministic conversation routing
- Two-stage LLM pipeline
- Explainable retrieval
- Catalog normalization
- Grounded recommendation generation
- Deadline-aware request budgeting
- Retry and graceful fallback mechanisms
- Independent evaluation framework
- Stateless API architecture

---

## Future Improvements

- Hybrid lexical + embedding retrieval
- Learning-to-rank recommendation models
- Adaptive clarification strategies
- Continuous evaluation dashboards
- Support for multiple assessment catalogs

---

## Acknowledgements

This project uses the **SHL Individual Test Solutions Catalog** as its knowledge base. All assessment names, product information, and related metadata remain the property of **SHL**.
