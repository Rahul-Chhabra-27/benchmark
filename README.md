# Qwen3-8B — LongBench Benchmark Results

This README tracks benchmark results for **Qwen3-8B** evaluated on the [LongBench](https://github.com/THUDM/LongBench) dataset across different memory budgets (e.g., KV-cache / GPU memory constraints).
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
| lcc | | 64.87| 64.87| 64.87| 64.87 |
| repobench-p | | 60.24| 60.25| 60.25| 60.25 |

## Notes

- Fill in each cell with the corresponding score once evaluation runs complete.
- The "Average" row can be computed as the mean across all 21 tasks (or split into English/Chinese/code subsets if desired).
- Memory columns (2GB/4GB/6GB/8GB) refer to the memory budget allocated for the model/KV-cache during inference — specify exactly what this constrains (e.g., KV-cache size, quantization level, GPU memory limit) here for clarity.

## How to Reproduce

```bash
# Example placeholder — adjust to your actual evaluation pipeline
python eval_longbench.py \
  --model Qwen3-8B \
  --tasks narrativeqa qasper multifieldqa_en multifieldqa_zh hotpotqa \
          2wikimqa musique dureader gov_report qmsum multi_news \
          vcsum trec triviaqa samsum lsht passage_count \
          passage_retrieval_en passage_retrieval_zh lcc repobench-p \
  --memory_budget <2GB|4GB|6GB|8GB>
```
