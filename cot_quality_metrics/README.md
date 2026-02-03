This project is designed to develop and test some novel metrics for the quality and faithfulness of Chain of Thought reasoning.
The idea is to treat the CoT as if it were a 'research log' in a notebook by a student in a lab class.
A bunch of ideas have been proposed in the planning doc. The goal is to try to make each of these into a rubric, score each seperately with a different LLM call, and then analyze the scores.
It may turn out that some of these are closely correlated in our observations, if so we may decide to merge those measures.
It may turn out that a human or LLM 'undirected vibe-check' of the quality of the CoT without rubric seems only weakly correlated with the aggregate score. If this is the case, then we may be missing key dimensions and need to brainstorm more qualities to divise rubrics around.
