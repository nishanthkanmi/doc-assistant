# Document Assistant: Q&A over SEC filings (a RAG pipeline with evaluation)

A document question-answering assistant. You ask a question in plain English
about long company reports, and it answers with citations to the passages it
used. It also tests itself: it scores its answers against 40 questions whose
correct answers are known.

It runs entirely on your own computer with free tools. There are no API keys
and no accounts.

## How this project was made

I built this project with AI-generated code. An AI coding assistant (Claude
Code) wrote the code, working step by step from prompts I gave it; I did not
write it line by line by hand. I then studied the code to understand how the
parts fit together (cleaning, chunking, indexing, retrieval, answering and
evaluation) so that I can explain it. The numbers in this README were produced
by the project's own evaluation script, run on a Windows 11 PC.

## What it does

The reports are the 10-K annual reports that Apple, Microsoft and Tesla filed
with the U.S. Securities and Exchange Commission (SEC). A 10-K runs to 100 pages
or more, with dense legal wording and hundreds of tables, so finding one number
in it by hand is slow.

The technique is called RAG ("retrieval-augmented generation"). The language
model is not asked to remember facts. Instead the pipeline first *retrieves* the
passages most relevant to a question, then asks the model to *generate* an
answer from only those passages. Because the answer is built from named
passages, you can check it against its source.

The pipeline has four stages: **clean** (turn web pages into plain text),
**chunk** (split the text into small pieces, keeping tables whole), **index**
(turn each chunk into numbers that capture its meaning, so it can be searched
by meaning) and **evaluate** (score the answers). At question time it
retrieves the 5 best chunks and, if the best one is a weak match, answers
"I don't know" instead of guessing.

## Example

    py src/ask.py "What were Apple's total net sales in 2025?"

Output (abridged; the source chunk is cut where marked):

    Question: What were Apple's total net sales in 2025?

    Answer (llama3.2):
    $416,161 million. [1]

    Sources (top 5 chunks):

    [1] AAPL-0153  score 0.878
        Apple Inc. 10-K, fiscal year ended September 27, 2025, Item 8. Financial Statements and Supplementary Data
        The following tables show information by reportable segment for 2025, 2024 and 2023 (in millions):
        ...
        Americas | Europe | Greater China | Japan | Rest of Asia Pacific | Corporate | Total
        Net sales | $178,353 | $111,032 | $64,377 | $28,703 | $33,696 | $— | $416,161
        ...

$416,161 million is the figure in Apple's report, and the SEC's structured data
agrees. A question the reports cannot answer is refused:

    py src/ask.py "What were Amazon's total net sales in 2025?"

    Answer (no model call: best score 0.749 is below min_score 0.76):
    I don't know. The filings do not appear to contain anything relevant to that question.

## Stack

- **Python 3** (written and run on Windows 11 with Python 3.15)
- **[Ollama](https://ollama.com)** runs two free models on your own computer:
  `nomic-embed-text` (turns text into numbers so it can be searched by meaning)
  and `llama3.2` (a small language model that writes the answers)
- **NumPy** for the similarity search, **BeautifulSoup** for cleaning web pages,
  **requests** for downloads, **ruamel.yaml** for the settings file
- **SEC EDGAR** (the filings) and the SEC's **XBRL "companyfacts"** data (the
  official answer key for the numeric test questions)

## Install and run

I have only run this on Windows 11. The commands use the Windows `py` launcher;
on macOS or Linux, use `python3` instead (untested).

1. **Install Python 3** from [python.org](https://www.python.org/downloads/).
2. **Install [Ollama](https://ollama.com)** and start it. You need about 2.3 GB
   of disk space for the two models.
3. **Download the two models** (once):

        ollama pull llama3.2
        ollama pull nomic-embed-text

   If the `ollama` command is not found, use PowerShell and Ollama's local
   web address instead:

        Invoke-RestMethod -Method Post http://localhost:11434/api/pull -Body '{"model": "llama3.2"}'
        Invoke-RestMethod -Method Post http://localhost:11434/api/pull -Body '{"model": "nomic-embed-text"}'

4. **Get the code** and install its Python packages:

        git clone https://github.com/nishanthkanmi/doc-assistant.git
        cd doc-assistant
        py -m pip install -r requirements.txt

5. **Download the filings** from the SEC (once). The SEC requires every
   automated request to say who is making it, so use your own email address:

        py src/download_filings.py --email you@example.com

6. **Run the whole pipeline** (clean, chunk, index, evaluate; about 5 minutes):

        py run_all.py

7. **Ask a question:**

        py src/ask.py "What were Apple's total net sales in 2025?"

**Reproducing the score.** The test set in `data/eval/questions.json` was built
for three specific filings: Apple `0000320193-25-000079` (fiscal year ended
September 27, 2025), Microsoft `0001193125-26-323660` (June 30, 2026) and Tesla
`0001628280-26-003952` (December 31, 2025). `download_filings.py` fetches each
company's *newest* 10-K, so if a company has filed a newer one since, the numeric
answers will no longer match. Rebuild the questions with
`py src/build_eval_questions.py --email you@example.com`. That regenerates the 25
numeric answers from SEC data, but the 10 text and 5 unanswerable questions were
written by hand for the filings above and may need updating.

**Every setting is in [config.yaml](config.yaml)**: the documents folder, chunk
size, embedding model, language model, number of chunks retrieved and the
abstention threshold.

To run on your own documents, change `documents.folder` (and `documents.workspace`,
so derived files stay separate), put a questions file at `evaluation.questions_file`
(format: see `data/eval/questions.json`), and run `py run_all.py`. Without a
questions file the evaluate stage is skipped.

## How the evaluation works

The test set has 40 questions across the three companies:

- **25 numeric questions** (for example "What was Microsoft's net income in fiscal
  year 2026?"). The correct answers come from the SEC's official structured XBRL
  data, not from the model or from this project's own text extraction, and each
  value is checked to appear in the cleaned filing. An answer is correct if a
  number in it is within 0.5% of the SEC value (diluted earnings per share must
  be within half a cent). "$416.2 billion" and "$416,161 million" both count.
- **10 text questions** with short reference answers (for example "What are
  Microsoft's three reportable segments?"). An answer is correct if it contains
  every key fact listed for the question.
- **5 questions the reports cannot answer** (for example Amazon's sales, or
  Microsoft's revenue in 2030). The correct behavior is "I don't know".

Every question goes through the same code as `ask.py`, so the test measures the
real program. Per-question results are saved in `data/eval/results.csv`.

## Results

**31 of 40 correct (77.5%)** with llama3.2, nomic-embed-text, the top 5 chunks
and an abstention threshold of 0.76.

| Category | Correct | Total | Accuracy |
|---|---|---|---|
| Numeric | 19 | 25 | 76.0% |
| Text | 8 | 10 | 80.0% |
| Unanswerable (should say "I don't know") | 4 | 5 | 80.0% |
| **Overall** | **31** | **40** | **77.5%** |

What this score does and does not show:

- Forty questions is a small test. One question is worth 2.5 percentage points,
  so a change of one or two questions can be chance.
- The abstention threshold (0.76) was chosen after looking at these same 40
  questions, so the "I don't know" result is optimistic. Similarity scores for
  answerable and unanswerable questions overlap, and the threshold caught only 2
  of the 5 unanswerable questions; the model itself declined 2 more.
- The scoring is lenient in places. For example, an answer is accepted if any
  number in it matches, and a bare number such as "416,161" is also tried as
  millions.
- Answer wording changes from run to run even at temperature 0, so expect
  scores to match closely, not exactly.
- Six of the nine misses are numeric answers where retrieval found the right
  area but the small model picked the wrong figure from a table (for example
  Tesla's cash plus short-term investments instead of cash). I have not yet
  tried a larger model or other retrieval methods, so I do not know how much
  they would help.
- It covers three filings from one fiscal year each, so it says little about
  how the system would do on other documents.

## Architecture

### The stages

    BUILD  (py run_all.py)

      documents folder        config.yaml  (every setting, one file)
      data/raw/*.html               |
            |                       | read by every stage
            v                       v
      +-----------+  text   +-----------+ chunks  +-----------+  index
      |  1 clean  | ------> |  2 chunk  | ------> |  3 index  | ------> data/index/
      |  Cleaner  |         |  Chunker  |         | Retriever |
      +-----------+         +-----------+         |  .build   |
            |                     |               +-----------+
       data/clean/*.txt    data/chunks/chunks.jsonl

    ASK  (py src/ask.py "question"; also the evaluate stage, for each test question)

      question --> Retriever.search --> top_k chunks, each with a score
                                              |
                              best score < abstention.min_score ?
                                   |                        |
                                  yes                       no
                                   v                        v
                        "I don't know."      LanguageModel.generate(prompt + chunks)
                                                            |
                                                            v
                                               answer, cited as [1] [2] ...

    EVALUATE  (stage 4 of run_all.py)

      questions.json --> ask each question --> score --> data/eval/results.csv
        25 numeric (answers from SEC XBRL data), 10 text, 5 unanswerable

Four parts are swappable: the **Cleaner**, **Chunker**, **Retriever** and
**Language model**. Each is a small class defined in `src/interfaces.py`; the
`type` setting of its section in `config.yaml` picks which one runs.

### What each file does

| File | Purpose |
|---|---|
| `config.yaml` | All settings. The only file you normally need to edit. |
| `run_all.py` | Runs the pipeline: `--stages clean,chunk,index,evaluate` (default all), `--config other.yaml` |
| `src/config.py` | Reads `config.yaml`; works out the derived folders (`clean/`, `chunks/`, `index/`, `eval/`) |
| `src/interfaces.py` | The four interfaces (`Cleaner`, `Chunker`, `Retriever`, `LanguageModel`) and what data passes between them |
| `src/registry.py` | Turns a `type` setting into a working component; lists the built-in names |
| `src/pipeline.py` | The stages themselves (clean, chunk, index) and `ask_question`, which does retrieve, abstain-or-answer |
| `src/evaluation.py` | Scores answers and writes `results.csv`; used by `run_all.py` and `src/evaluate.py` |
| `src/components/cleaners.py` | `html` (removes scripts, hidden tags and XBRL data, keeps tables as rows), `text` |
| `src/components/chunkers.py` | `sec_filing` (splits on Item sections, keeps table rows together, drops page headers, realigns shifted table headings), `paragraph` (any text) |
| `src/components/retrievers.py` | `ollama_embeddings` (finds chunks by meaning), `tfidf` (finds chunks by shared words, no model needed) |
| `src/components/language_models.py` | `ollama` (any chat model served by local Ollama) |
| `src/components/ollama.py` | The one function that talks to Ollama's HTTP API |
| `src/ask.py` | Ask one question; prints the answer and the source chunks |
| `src/evaluate.py` | Run only the evaluation (`--category numeric` for one category) |
| `src/download_filings.py` | Fetch the 10-Ks listed in `download.tickers` from SEC EDGAR |
| `src/build_eval_questions.py` | Build the 40-question test set; numeric answers come from SEC XBRL data |
| `data/raw`, `clean`, `chunks`, `index`, `eval` | Documents, then each stage's output |

### How to swap a component

Change the `type` line of that section in `config.yaml`, then re-run the stages
that come after it. Built-in types are listed in the comments of `config.yaml`.

| To change | Edit in `config.yaml` | Then run |
|---|---|---|
| Chunker | `chunker.type: paragraph` | `py run_all.py --stages chunk,index,evaluate` |
| Retriever | `retriever.type: tfidf` (and set `abstention.min_score` for its score scale) | `py run_all.py --stages index,evaluate` |
| Language model | `language_model.name: some-other-model` (any model pulled into Ollama) | `py run_all.py --stages evaluate` |
| Embedding model | `retriever.embedding_model` and its prefixes | `py run_all.py --stages index,evaluate` |
| Cleaner | `cleaner.type: text` (for `.txt` and `.md` sources) | `py run_all.py` |
| Settings only | chunk size, `top_k`, `min_score` | stages after the one that uses it |

Scores are not comparable between retrievers: `ollama_embeddings` scores run
about 0.75 to 0.89 here, while `tfidf` scores are much lower, so
`abstention.min_score` must be chosen for the retriever in use.

**Writing your own component.** Subclass the interface, then use
`module.path:ClassName` as the `type`. The module must be importable from the
project folder or `src/`. For example, a language model that calls a hosted
service:

    # src/my_models.py
    from interfaces import LanguageModel

    class HostedModel(LanguageModel):
        def generate(self, system_prompt, user_prompt):
            name = self.options["name"]      # your section of config.yaml
            ...                              # call your service, return the reply text

    # config.yaml
    language_model:
      type: my_models:HostedModel
      name: whatever-it-needs

A component is created as `YourClass(options, config)`, where `options` is its own
section of `config.yaml` (so you can add any settings you like) and `config` is
the whole file. The other interfaces are: `Cleaner.clean(raw_bytes) -> text`
(plus an `extensions` list), `Chunker.chunk(text, doc_name) -> list of chunk dicts`,
and `Retriever.build(chunks)` / `.load()` / `.search(question, top_k)`.
A chunk dict needs `id`, `text` and `heading`; a search result adds `score`.

**Limits to know about.** There is no PDF cleaner yet (add one as a `Cleaner`).
`sec_filing` and `build_eval_questions.py` are specific to SEC 10-K filings.
A folder is cleaned by one cleaner, so mix file types only if you write a cleaner
that handles them.

## Re-running the evaluation

(How the scoring works is described under "How the evaluation works" and
"Results" near the top.)

    py src/build_eval_questions.py --email you@example.com   # rebuild data/eval/questions.json (optional: it is included)
    py run_all.py --stages evaluate                          # about 3 minutes
    py src/evaluate.py --category numeric                    # one category only, for a quick check

One row per question is written to `data/eval/results.csv`, and accuracy is
printed per category. `abstention.min_score` in `config.yaml` makes the pipeline
answer "I don't know" when the best chunk is not similar enough to the question.

## Changelog

- v0.1: project created, step 1 (load) written.
- v0.2: step 2 (clean) written.
- v0.3: steps 3 to 6 written. Chunking keeps table rows together, splits on
  Item sections, removes page headers and footers, and realigns shifted table
  headings. Indexing uses nomic-embed-text and retrieval takes the top 5 chunks
  by cosine similarity (a measure of how close two meanings are). Answers come
  from llama3.2 through Ollama's HTTP API; the model name is set in the config
  file. `ask.py` runs retrieve and answer together. Step 2 now also fixes
  `$ | %` heading cells and line breaks inside table cells.
- v0.4: step 7 (evaluate) written. 40 questions, answers for the numeric ones
  come from SEC XBRL data. Added `min_score` abstention (default 0.76).
  **Baseline** (llama3.2, nomic-embed-text, top 5 chunks, min_score 0.76):

  | Category | Correct | Total | Accuracy |
  |---|---|---|---|
  | numeric | 19 | 25 | 76.0% |
  | text | 8 | 10 | 80.0% |
  | unanswerable (abstained) | 4 | 5 | 80.0% |
  | overall | 31 | 40 | 77.5% |

  Notes on the baseline: the score threshold refused 2 of the 5 unanswerable
  questions and 0 of the 35 answerable ones; the model itself declined 2 more
  and wrongly answered "How many iPhone units did Apple sell?" with iPhone net
  sales. Similarity scores overlap heavily (0.78 to 0.89 for answerable
  questions, 0.75 to 0.83 for unanswerable), and 0.76 was chosen after looking
  at these same 40 questions, so treat the abstention result as optimistic.
  Most numeric misses were the model picking the wrong figure from a retrieved
  table (for example Tesla's cash plus short-term investments instead of cash).
- **v1.0: reusable foundation.** One `config.yaml` replaces `config.toml` and holds
  every setting. `py run_all.py` runs clean, chunk, index and evaluate on the
  configured documents folder. Cleaner, chunker, retriever and language model are
  swappable behind interfaces (`src/interfaces.py`) by changing a `type` setting
  or pointing it at your own `module:Class`; two built-in options each for the
  first three, so the swap is exercised. The per-step scripts (`clean_filings.py`,
  `chunk_filings.py`, `build_index.py`, `retrieve.py`, `answer.py`) became components.
  Adds the Architecture section above. New dependency: `ruamel.yaml`
  (PyYAML did not install on Python 3.15).
  **Evaluation scores, unchanged by the refactor** (llama3.2, nomic-embed-text,
  sec_filing chunker, top 5, min_score 0.76):

  | Category | Correct | Total | Accuracy |
  |---|---|---|---|
  | numeric | 19 | 25 | 76.0% |
  | text | 8 | 10 | 80.0% |
  | unanswerable (abstained) | 4 | 5 | 80.0% |
  | overall | 31 | 40 | 77.5% |

  Checked against the pre-refactor outputs: cleaned text, all 839 chunks and the
  embeddings are identical, and every question passes or fails exactly as before.
  The wording of a few answers changes from run to run (Ollama is not perfectly
  repeatable even at temperature 0), so expect scores to match closely, not
  always to the last question.
