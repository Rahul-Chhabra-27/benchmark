# Qwen3-8B — LongBench Benchmark Results

This README tracks benchmark results for **Qwen3-8B** evaluated on the [LongBench](https://github.com/THUDM/LongBench) dataset across different memory budgets (e.g., KV-cache / GPU memory constraints).

See [Ruler-32k-README.md](Ruler-32k-README.md) for the separate Qwen3-8B RULER-32K results table.

See [Loft-Qwen3-8B-README.md](Loft-Qwen3-8B-README.md) for the combined Qwen3-8B LOFT 32K and 128K results.

# LongBench statistics

| Task          | Task Type | Eval metric |     Avg len                            |Language | \#Sample |
| :-------- | :-----------:| :-----------: |:-------: | :-----------: |:--------: |
| HotpotQA   | Multi-doc QA | F1                        |9,151                           |EN                           |200                           |
| 2WikiMultihopQA| Multi-doc QA | F1                        |4,887                           |EN                           |200                           |
| MuSiQue| Multi-doc QA | F1                        |11,214                           |EN                           |200                           |
| DuReader| Multi-doc QA | Rouge-L                 |15,768                           |ZH                           |200                           |
| MultiFieldQA-en| Single-doc QA | F1                        |4,559                           |EN                           |150                           |
| MultiFieldQA-zh| Single-doc QA | F1                        |6,701                           |ZH                           |200                           |
| NarrativeQA| Single-doc QA | F1                        |18,409                           |EN                           |200                           |
| Qasper| Single-doc QA | F1                        |3,619                           |EN                           |200                           |
| GovReport| Summarization | Rouge-L                 |8,734                           |EN                           |200                           |
| QMSum| Summarization | Rouge-L                 |10,614                           |EN                           |200                           |
| MultiNews| Summarization  | Rouge-L                 |2,113                           |EN                          |200                           |
| VCSUM| Summarization | Rouge-L                 |15,380                           |ZH                           |200                           |
| TriviaQA| Few shot  | F1                        |8,209                           |EN                           |200                           |
| SAMSum| Few shot | Rouge-L                        |6,258                           |EN                           |200                           |
| TREC| Few shot | Accuracy                |5,177                           |EN                           |200                           |
| LSHT| Few shot | Accuracy                |22,337                           |ZH                           |200                           |
| PassageRetrieval-en| Synthetic | Accuracy                |9,289                           |EN                           |200                           |
| PassageCount| Synthetic | Accuracy                |11,141                           |EN                           |200  |
| PassageRetrieval-zh | Synthetic | Accuracy                |6,745                           |ZH                           |200                           |
| LCC| Code | Edit Sim              |1,235                           |Python/C#/Java                           |500                           |
| RepoBench-P| Code | Edit Sim                |4,206                           |Python/Java                           |500                           |


# Task description

| Task              | Task Description                                            |
| :---------------- | :----------------------------------------------------------- |
| HotpotQA          | Answer related questions based on multiple given documents   |
| 2WikiMultihopQA   | Answer related questions based on multiple given documents   |
| MuSiQue           | Answer related questions based on multiple given documents   |
| DuReader          | Answer related Chinese questions based on multiple retrieved documents |
| MultiFieldQA-en   | Answer English questions based on a long article, which comes from a relatively diverse field |
| MultiFieldQA-zh   | Answer Chinese questions based on a long article, which comes from a relatively diverse field |
| NarrativeQA       | Answer questions based on stories or scripts, including understanding of important elements such as characters, plots, themes, etc. |
| Qasper            | Answer questions based on a NLP research paper, questions proposed and answered by NLP practitioners |
| GovReport         | A summarization task that requires summarizing government work reports |
| MultiNews             | A multi-doc summarization that requires summarizing over multiple news   |
| QMSum             | A summarization task that requires summarizing meeting records based on user queries |
| VCSUM             | A summarization task that requires summarizing Chinese meeting records |
| SAMSum            | A dialogue summarization task, providing several few-shot examples                    |
| TriviaQA          | Single document question answering task, providing several few-shot examples |
| NQ                | Single document question answering task, providing several few-shot examples |
| TREC              | A classification task that requires categorizing questions, includes 50 categories in total |
| LSHT              | A Chinese classification task that requires categorizing news, includes 24 categories in total |
| PassageRetrieval-en | Given 30 English Wikipedia paragraphs, determine which paragraph the given summary corresponds to |
| PassageCount | Determine the total number of different paragraphs in a given repetitive article |
| PassageRetrieval-zh | Given several Chinese paragraphs from the C4 data set, determine which paragraph the given abstract corresponds to |
| LCC               | Given a long piece of code, predict the next line of code |
| RepoBench-P       | Given code in multiple files within a GitHub repository (including cross-file dependencies), predict the next line of code |

## Setup

- **Model:** Qwen3-8B
- **Benchmark:** LongBench
- **Memory configurations tested:** 2GB, 4GB, 6GB, 8GB
- **Metric:** *(fill in — e.g., F1 / Rouge-L / Accuracy, as defined per-task by LongBench)*

## Results
## Meta-llama/Llama3.1-8B-Instruct
| Task | 256MB | 512MB | 768MB | 1GB | 2GB | 4GB | 6GB | 8GB | Average |
|---|---|---|---|---|---|---|---|---|---|
| narrativeqa | 24.85 | 30.78 | 30.77 | 32.14| 30.07| 30.81| 30.66|30.62 | 30.86 |
| qasper | 45.4 | 48.43 | 47.22 | 47.46| 47.15| 47.25| 47.25|47.25 | 47.27 |
| multifieldqa_en | 56.06 | 56.54 | 56.22 |56.48 |54.95 | 54.95| 54.95| 54.95| 55.26 |
| multifieldqa_zh | 61.58 | 62.99 | 63.38 | 63.46| 63.49| 63.49| 63.49|63.49 | 63.48 |
| hotpotqa | 58.0 | 54.26 | 54.44 | 57.55| 59.32| 59.27| 59.27| 59.27| 58.94 |
| 2wikimqa | 48.34 | 48.98 | 50.58 | 51.37 | 51.09| 51.09|51.09 | 51.09| 51.15 |
| musique | 28.49 | 25.68 | 28.23 | 29.57| 32.76| 32.76| 32.76| 32.76| 32.12 |
| dureader | 31.27 | 32.78 | 32.87 |32.69 | 32.28| 32.36|32.36 | 32.36| 32.41 |
| gov_report | 20.11 | 20.37 | 20.43 |20.33 | 20.34| 20.38| 20.35| 20.35| 20.35 |
| qmsum | 24.35 | 25.17 | 25.17 | 25.17| 24.92| 24.86| 24.86|24.86 | 24.91 |
| multi_news | - | - | - | 22.0| 21.97| 21.97| 21.97| 21.97| 21.98 |
| vcsum | - | - | - |14.67 | 14.8| 14.63| 14.63| 14.63| 14.67 |
| trec | - | - | - | 27.0| 29.5| 29.5| 29.5| 29.5| 29.00 |
| triviaqa | - | - | - | 90.83| 91.71| 92.21| 92.21| 92.21| 91.83 |
| samsum | - | - | - | 37.96|40.88| 40.85| 40.85| 40.85| 40.28 |
| lsht | - | - | - | | | | | | |
| passage_count | - | - | - |11.61 |11.42 | 12.17| 12.17|12.17 | 11.91 |
| passage_retrieval_en | - | - | - | 100.0| 100.0| 100.0|100.0 |100.0 |100.0 |
| passage_retrieval_zh | - | - | - | 99.0|99.0 | 99.0| 99.0| 99.0| 99.0|
| lcc | - | - | - | 51.15| 51.12| 51.06| 51.06| 51.06| 51.09 |
| repobench-p | - | - | - |44.77 |44.19 |44.51 | 44.51|44.51 | 44.50 |

## Qwen3-8B
| Task | 2GB | 4GB | 6GB | 8GB | Average |
|---|---|---|---|---|---|
| narrativeqa | 27.85| 28.80| 28.87| 28.87| 28.60 |
| qasper | 43.67| 43.74| 43.74| 43.74| 43.72 |
| multifieldqa_en |55.63 |55.63 | 55.63|55.63 | 55.63 |
| multifieldqa_zh | 66.07| 66.07|66.07 | 66.07| 66.07 |
| hotpotqa | 62.8| 62.8| 62.8|62.8 | 62.80 |
| 2wikimqa |48.3 | 48.3| 48.3|48.3 | 48.30 |
| musique | 34.90| 34.97| 34.97| 34.97| 34.95 |
| dureader | 26.76| 26.83| 26.83| 26.83| 26.81 |
| gov_report |18.3 | 18.3| 18.3| 18.3| 18.30 |
| qmsum |24.48 | 24.62| 24.64| 24.64| 24.60 |
| multi_news | 19.9| 19.9| 19.9|19.9| 19.90 |
| vcsum | 13.48| 13.5| 13.5| 13.5| 13.50 |
| trec | 40.5| 40.5| 40.5|40.5 | 40.50 |
| triviaqa | 90.46| 90.46| 90.46| 90.46| 90.46 |
| samsum | 40.4| 40.57| 40.57| 40.57| 40.53 |
| lsht |20.0 |20.0 | 20.0| 20.0| 20.00 |
| passage_count | 9.0| 10.0|10.0 |10.0 | 9.75 |
| passage_retrieval_en |91.87 | 91.87|91.87 |91.87 | 91.87 |
| passage_retrieval_zh | 99.0| 99.0| 99.0| 99.0| 99.00 |
| lcc | 64.87| 64.87| 64.87| 64.87| 64.87 |
| repobench-p | 60.41| 60.24| 60.25| 60.25| 60.25 |
