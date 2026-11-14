# llm-judge-lab

How much of an LLM judge's score is the judge, and how much is the text?

Three local models (llama3.1:8b, qwen2.5:7b, dolphin-mistral) each write system
prompts, rewrite customer-support answers under those prompts, and grade **every**
rewrite, their own included. Because all three judges see the same 270 texts, judge
behaviour can be separated from answer quality.

The repo also contains a staged generator of labelled synthetic support dialogues,
whose stylization step picks a (prompt-writer, actor) model pair per situation with a
small judged tournament.

## Findings

From one run (2026-03-02, 5 Bitext requests, 270 rewrites, 810 verdicts). Every number
below is regenerated from the raw JSONL by `experiments/analyze_cross_judge.py`
([full report](results/cross_judge_2026-03-02/REPORT.md)) and pinned by
`tests/test_analysis.py`.

**1. A safety refusal scored as 0 is a measurement bug.** llama3.1 declined to grade
116 of 270 texts, 114 of them from the deliberately toxic persona ("I can't evaluate
this dialogue because it contains offensive statements", translated from Ukrainian). The original pipeline stored every
unreadable verdict as `score = 0`, so llama looked like the harshest judge (mean 2.51).
On the texts it actually graded, its mean is 4.56.

**2. The judge still matters more than it should.** On the 149 texts all three judges
graded, mean scores are 7.45 (dolphin-mistral), 6.43 (qwen2.5) and 4.56 (llama3.1): a
2.89-point spread on a 10-point scale for identical inputs.

**3. The judges barely agree.** Pairwise Spearman rank correlation is 0.25–0.43. All
three reach the same pass/fail decision on 27 of those 149 texts (18%).

**4. A judge's own pass/fail flag contradicts its own score.** Asked for
`{"score", "accepted"}` with a pass mark of 7, qwen2.5 returned an `accepted` flag that
disagreed with its score 112 times out of 270. By its flag it passed 6% of texts; by
its scores, 47%. Pass/fail is now derived from the score and never read from the model.

**5. One judge prefers its own writing.** Comparing each judge with the other two on
the same text, qwen2.5 rates its own rewrites +0.84 points higher than its usual offset
(bootstrap 95% CI [+0.43, +1.26]). For llama3.1 and dolphin-mistral the interval
includes zero.

**6. The rubric's language leaks.** The rubric was written in Ukrainian for English
texts; dolphin-mistral answered in Russian in 67 of its 270 verdicts.

### Earlier runs

Before the cross-judge design, a single judge (qwen2.5) graded a single generator
(llama3.1) ([report](results/earlier_runs/REPORT.md)):

- Restyling the dataset's reference answer passed 20/20 versus 15/20 for writing from
  scratch, but every accepted text in both arms scored exactly the pass mark (8). With
  a refine-until-pass loop the judge acted as a binary gate, so the means carry little
  information beyond the pass count.
- Letting qwen2.5 rewrite the system prompts after seeing the baseline's failures made
  things worse: 16/50 accepted versus 22/50 with hand-written prompts. The toxic persona
  stayed at 0/10 either way.

Those results are why the cross-judge run grades every text with every model.

## What changed in the code because of this

| problem found in the data | change |
|---|---|
| refusals and truncated outputs stored as 0 | `judgelab.verdict` returns `scored`, `refused` or `unparsed`; analyses never average the last two as scores |
| model's `accepted` flag contradicts its score | `passed` is computed from the score; the flag is kept only to measure the contradiction |
| 5 verdicts were prose with no JSON, under a 128-token cap | cap raised to 256 so a model that reasons first can still reach the JSON |
| Ukrainian rubric on English texts | English prompt set is the default; the original Ukrainian set is kept verbatim (`--language uk`) to reproduce the 2026-03-02 run |
| micro-arena scored an unreadable verdict as 5.0 | unreadable candidates are dropped |
| "epsilon-greedy" router only counted, never explored | exploration is real and deferred to between dialogues; the registry keeps the better of old and new route |
| three different LLM clients, one with a hand-written chat template | one `Backend` protocol over Ollama's `/api/chat`, plus a scripted backend for tests |

LLM calls are network-bound, so the experiment runs them in a `ThreadPoolExecutor`:
threads release the GIL while waiting on the socket, and throughput is limited by the
Ollama server rather than by Python.

## Limitations

- **Small and single-shot.** Five source requests, three 7–8B models, one sampling run
  at non-zero temperature. The numbers describe this run; they are not estimates for
  "LLM judges" in general.
- **No human labels.** The data shows that judges disagree, not which one is right.
- **Confounded subset.** llama3.1 refused most toxic texts, so the 149 texts every judge
  graded are 133 cheerful and 16 toxic.
- **Dependent samples.** Bootstrap intervals treat generations as independent, but
  they share source requests and system prompts; the intervals are optimistic.
- **Pattern-based refusal detection.** `is_refusal` matches refusal phrases in English,
  Ukrainian and Russian. It was checked against every zero in this run, not against a
  labelled set.

## The dialogue generator

`judgelab.dialogues` produces customer-support chats with ground-truth labels (intent,
case type, satisfaction, agent mistakes, hidden dissatisfaction), so the corpus can be
used to test a dialogue analyzer against known answers.

```
seed -> dice (hidden labels) -> scenario -> beat sheet -> draft -> stylize -> macro judge
                                               ^                                   |
                                               +------------ retry on fail --------+
```

- The **beat sheet** separates *what happens* from *when it is revealed*: each message
  is drafted from its own beat and the last three messages, so the drafter cannot
  resolve the problem in turn 2.
- **Stylization** looks up a cached route (prompt-writer model, actor model, system
  prompt) for the context (client or agent style, satisfaction, case type). On a miss,
  a micro-arena has every prompt-writer write a prompt, every actor rewrite sample
  messages under it, and a judge pick the winner, which is cached in SQLite.
- The **macro judge** checks the whole chat for spoilers, flow, format, and whether the
  planted mistakes and hidden dissatisfaction actually appear; an unreadable verdict
  fails the gate.

I wrote the first version during the INT20H hackathon (team submission:
[ValeriiaBaranivska/EDLWSS-INT20H-AI](https://github.com/ValeriiaBaranivska/EDLWSS-INT20H-AI))
and rebuilt it afterwards into this staged design. The ported stages are tested on a
scripted backend; this repo does not yet contain a quantitative evaluation of the
generated dialogues.

## Reproduce

```bash
pip install -e ".[dev]"
pytest                                   # 42 tests, no Ollama needed

# regenerate the reports from the committed raw data
python experiments/analyze_cross_judge.py results/cross_judge_2026-03-02
python experiments/summarize_earlier_runs.py results/earlier_runs

# new run (needs Ollama and the three models)
ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull dolphin-mistral
pip install -e ".[data]"
python experiments/run_cross_judge.py --records 5 --language en
python experiments/generate_dialogues.py --n 10 --seed 42
```

## Layout

```
src/judgelab/
  llm.py              Backend protocol, Ollama client, scripted test backend
  verdict.py          judge output -> scored / refused / unparsed
  data.py             Bitext seed records (downloaded on demand)
  cross_judge/        prompts (en + original uk), experiment, analysis
  dialogues/          dice, planning, drafting, stylize, macro judge, pipeline
    routing/          registry (SQLite), epsilon-greedy router, micro-arena
experiments/          one script per question
results/              raw JSONL of past runs + generated reports
tests/
```

## Data

Source requests come from the
[Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(CDLA-Sharing-1.0, © Bitext Innovations 2024). The dataset is not stored here; files in
`results/` contain a few of its records quoted alongside model outputs, under the same
licence. Code is MIT-licensed.
