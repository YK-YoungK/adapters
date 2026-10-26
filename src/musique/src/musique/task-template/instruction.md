# MuSiQue question answering

Read `/workspace/question.json`. It contains a question and its candidate paragraphs,
each with an `idx`, `title`, and `paragraph_text`. Answer the question using these
paragraphs, and identify all paragraph indices that support your answer. All
questions in this MuSiQue-Ans split are answerable from the supplied paragraphs.

Write one JSON object to `/workspace/prediction.json` with these fields:

- `id`: the exact question ID from `question.json`.
- `predicted_answer`: a string containing only the short answer, without explanation.
- `predicted_support_idxs`: a list of integer paragraph indices supporting the answer.
- `predicted_answerable`: the boolean `true`.

Use the original paragraph indices, not their titles or a new numbering scheme.
Answer quality and supporting-paragraph quality are scored separately using the
official MuSiQue metrics. Your final chat message is not read by the evaluator;
save the JSON file at the specified path.
