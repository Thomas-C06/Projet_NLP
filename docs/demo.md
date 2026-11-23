# Presentation and demonstration

All group members should explain their own contribution. The presentation can follow this order:

1. State the user need and show an English request with several contextual expectations.
2. Describe the 150-film corpus, its provenance, extracted attributes and estimated labels. Do not claim an independent manual review that was not done.
3. Explain how text becomes embeddings, then how cosine similarities become an affinity index.
4. Show the questionnaire and submit a request with themes and atmosphere.
5. Inspect a result, its evidence, weights and contributions, then the radar chart of the match profile. Explain that affinity is not a probability.
6. Switch the comparison method to "Keywords only (TF-IDF baseline)" with the same request and show how the selection changes without semantics.
7. Add a strict era, duration or genre filter and show the changed selection.
8. Show fewer than three results and explain why filters are kept.
9. Tick "Save my answers to the local study log" and open `data/responses/responses.jsonl` to show the structured record.
10. If Ollama is running, press "Ask the critic" and explain that the LLM only rephrases the computed match (bonus feature, not used for ranking).
11. Present the three-method results measured with automatic LLM judgments, the scoring configuration selected on validation queries, and difficult cases. Say clearly that the judge is an LLM, not a human panel.
12. Discuss missing annotations, language detection errors, negations and limits of the test set.

Before class, follow the setup in the README to build the frontend and prepare the local model. Start the app with Uvicorn, open it at http://127.0.0.1:8000, then disconnect the network and try a recommendation. Keep the PDF documentation and technical check outputs available locally.

Do not present technical unit tests as proof of recommendation relevance. Relevance requires the independent human judgments described in the evaluation guide.
