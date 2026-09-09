# QA Test Set

Fill this in with real questions/answers from the 2-3 sample syllabi in `data/sample_syllabi/`.
Run each through the app and check the answer against "Expected" — flag mismatches.

| # | Question | Expected Answer | Actual Answer | Pass/Fail |
|---|---|---|---|---|
| 1 | When is the midterm exam? | | | |
| 2 | What % of the grade is the final exam? | | | |
| 3 | What textbook is required? | | | |
| 4 | What are the professor's office hours? | | | |
| 5 | Is attendance mandatory? | | | |
| 6 | What is the late submission policy? | | | |
| 7 | How many assignments are there? | | | |
| 8 | What topics are covered in week 3? | | | |
| 9 | Is there a group project? | | | |
| 10 | (Ask something NOT in the syllabus — should say "not found") | I couldn't find that in the syllabus. | | |

## Setup Checklist
- [ ] Ollama running: `ollama serve` (and `ollama pull llama3` done ahead of time)
- [ ] Backend running: `uvicorn main:app --reload --port 8000` (from `backend/`)
- [ ] Frontend opened: `frontend/index.html` in browser
- [ ] 2-3 real syllabus PDFs placed in `data/sample_syllabi/`
