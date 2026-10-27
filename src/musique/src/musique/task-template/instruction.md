# MuSiQue question answering

Answer the question using the candidate paragraphs below, and identify all paragraph
indices that support your answer. Preserve the original paragraph indices.

{answerability_instruction}

Write one JSON object to `/workspace/prediction.json` with these fields:

- `id`: the exact question ID provided below.
- `predicted_answer`: a string containing the short answer, without explanation.
- `predicted_support_idxs`: a list of integer supporting paragraph indices.
- `predicted_answerable`: a boolean indicating whether the evidence is sufficient.

For insufficient evidence, you may use an empty answer and an empty support list.
Your final chat message is not read by the evaluator; save the JSON file at the
specified path.

The question and paragraphs are provided as JSON, in their original order:

```json
{question_json}
```
