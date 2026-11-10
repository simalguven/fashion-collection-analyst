# Fashion collection analyst

A small demo that turns a folder of product photos into style groups and colour palettes, then lets you ask questions about them in plain English through a tested AI agent.



![Style clusters, typical products and dominant colours](reports/figures/cluster_gallery.png)

## Findings and recommendations

*This section is written for readers without a technical background. It covers a random sample of 1,999 products from a public catalogue of 44,072.*

### What the analysis found

- **The collection falls into 12 visual style groups, without anyone labelling them.** Nine are clear, such as watches, sunglasses, men's T-shirts and sports shoes. Three are mixed and hold 35% of all products, for example belts, wallets and socks grouped together.
- **The range is safe and casual.** 78% of products are casual wear and half are apparel. Men's and women's products make up 50% and 42%.
- **Bags are the most colourful category.** About half of handbags and backpacks (52%) have a main colour that is not black, white, grey or silver. For watches (15%) and sports shoes (16%), it is far fewer.
- **The catalogue is out of date.** 71% of its products date from 2011 to 2013, so it describes an old market and says nothing about current trends.
- **Automatic colour detection is useful but not yet reliable.** It picks the right colour family for 52% of products. Always guessing the most common family gets 21%, and analysing the photo without removing the background gets 17%.

### What I recommend

| Recommendation | Why |
|---|---|
| **Use the style groups and "find similar" for browsing and range reviews now.** | The groups broadly follow how the catalogue is organised, and they need no manual tagging. |
| **Treat colour results as a direction, and have a person check them before a buying or design decision.** | Nearly half of individual products get the wrong colour family. Pale blues and greens are often read as white, so neutral colours look more common than they are (63% detected, 46% in the catalogue labels). |
| **Have a merchandiser review the three mixed groups, or split them further.** | They hold a third of the products, and a single group label hides what is in them. |
| **Connect recent product data and sales figures before using this for trend decisions.** | This dataset has neither recent products nor sales. The agent correctly refuses to answer those questions today. |
| **Run the agent test suite before every change to the agent.** | The tests caught a crash and an unrequested colour substitution during this project, both easy to miss in a quick manual check. |

### How far you can trust the agent

The agent passed **39 of 40 test runs** (20 questions, each asked twice). These include missing products, questions outside its data, and attempts to trick it into stating something false. In these tests it looked up its answers and did not invent sales figures or colours.

The sample is small, and I improved the agent and the tests using these same questions, so this shows it handles these situations and is not a guarantee of general accuracy.

## What it does

1. **Data quality check** on the catalogue's labels and on the sampled images, removing unusable images.
2. **Image embeddings** from DINOv2 and CLIP, using Hugging Face `transformers`.
3. **Style clustering** with k-means, plus a UMAP map of the collection.
4. **Colour extraction** with OpenCV: background and skin removal, then k-means in CIELAB for each product.
5. **A LangGraph agent** with seven read-only tools, running on a Groq-hosted model through LangChain.
6. **Agent evaluation** with deterministic grading, where expected answers are computed from the data.

![Map of the collection](reports/figures/style_map.png)

## Technical decisions and results

**Dataset.** [`ashraq/fashion-product-images-small`](https://huggingface.co/datasets/ashraq/fashion-product-images-small) replaced Fashion-MNIST because Fashion-MNIST is greyscale, so it cannot support colour analysis. Its colour and category labels also make the results measurable.

**Embedding model.** Both models were scored on agreement with article types, which clustering never sees. DINOv2 was better, so it drives clustering and image similarity. CLIP is kept for text search, since it also embeds text. Full results are in [backbone_comparison.csv](reports/backbone_comparison.csv).

| 12 clusters | Silhouette | NMI | ARI |
|---|---|---|---|
| DINOv2 | 0.298 | 0.670 | 0.413 |
| CLIP | 0.209 | 0.642 | 0.358 |

**Number of clusters.** k is the smallest value within 5% of the best silhouette score, which gives 12. Silhouette is nearly flat from 12 to 18 and peaks at the edge of the scanned range (20), so taking its maximum would mostly add clusters that are harder to review.

**Colour accuracy.** Measured on colour families (12 groups such as Blue or Red), on the half of the items not used for tuning. I looked at that half more than once during development, so treat the numbers as slightly optimistic.

| Method | Accuracy |
|---|---|
| Background and skin removal, CIELAB k-means | 51.7% |
| Always predict the most common family | 21.2% |
| Same k-means on the whole image | 16.7% |

The first version scored 39.8% because OpenCV's 8-bit LAB format overweights lightness. Using true CIELAB values and handling on-model skin tones fixed most of that gap.

**Data quality.** Details are in [data_quality.json](reports/data_quality.json).

| Check | Result |
|---|---|
| Missing key labels (44,072 rows) | None |
| Repeated product names | 13,271 (30%) |
| Article types with fewer than 10 items | 37 of 141 |
| Products from 2011 to 2013 | 71% |
| Image size | 60x80 pixels for 99.9% of the sample |
| Sample images unreadable, blank | 0, 0 |
| Sample exact duplicates (removed), near duplicates (kept) | 1, 31 |

## Agent evaluation

The 20 test questions are in [evals/cases.py](evals/cases.py). A run passes only if the agent called the required tools, avoided forbidden ones, and gave an answer that matches values computed from the data.

| Category | First run | Final run (each asked twice) |
|---|---|---|
| Happy path | 7/7 | 14/14 |
| Multi-step | 3/3 | 6/6 |
| Edge cases | 2/5 | 9/10 |
| Out of scope | 3/3 | 6/6 |
| Adversarial | 1/2 | 4/4 |
| **Total** | **16/20** | **39/40** |

Most of the improvement came from correcting the tests, not from a better agent. Only one of the first four failures was a real agent fault.

**Failure notes**

| Case | What happened | Type | Outcome |
|---|---|---|---|
| `unknown_colour` (first run) | The model emitted a malformed tool name. Groq rejected the request and the whole run crashed. | Agent robustness | Fixed with a corrective retry, covered by a unit test. |
| `unknown_colour` (final run) | Asked about "ultraviolet", the agent said it isn't tracked, then answered for Purple anyway. It disclosed this, but it did not stop and ask. | Agent behaviour | **Open.** Kept as a failure and the check was not loosened. |
| `false_premise` | "Isn't mostly neon green" was marked wrong because the grader did not recognise "isn't". | Test bug | Fixed. |
| `limit_too_high` | The agent correctly explained the limit of 20 from the tool description, but the test required a tool call. | Test too strict | Fixed. |
| `ambiguous_reference` | The agent listed the clusters and asked which one was meant, which is helpful, but the test forbade any tool. | Test too strict | Listing is now allowed. |
| Lenient checks that passed | The prompt-injection check only rejected one exact phrase. | Test too weak | Tightened, with tests confirming that fabricated answers fail. |

Reports: [first run](reports/eval_report_baseline.md), [final run](reports/eval_report_final.md). One full 40-run evaluation used about 89,000 tokens.

**Known limits of the evaluation.** It is 20 questions on one model, and the same questions guided the fixes. Keyword-based grading can't judge the quality of wording. Colour answers are checked against the pipeline's own output, so a correct agent can still repeat a wrong colour detection.

## How to run

Requires Python 3.12. It runs on CPU, or faster on Apple Silicon or a GPU. The full pipeline took under 3 minutes on an M2 MacBook.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py     # downloads data, builds clusters, palettes and figures
pytest tests                       # no API key needed

cp .env.example .env               # then add a free key from console.groq.com
python evals/run_eval.py --label myrun --repeats 2
```

Then open [notebooks/collection_analyst.ipynb](notebooks/collection_analyst.ipynb) for the walkthrough.

To use another model, change `AGENT_MODEL` in `.env` and install its LangChain integration. Only the Groq model `qwen/qwen3.8-27b` has been tested.

## Project structure

```
src/fashion_analyst/   data, quality, embeddings, clustering, colours, plots, tools, agent
scripts/run_pipeline.py  builds every artifact
evals/                 test questions, runner and grading
tests/                 unit tests, including grader checks (no LLM needed)
notebooks/             walkthrough with outputs
reports/               figures, metrics and evaluation reports
```

## What I would improve next

- **Replace the skin and background heuristics with garment segmentation**, the biggest lever for colour accuracy, and use higher-resolution images than 60x80.
- **Split mixed clusters** with a second clustering pass inside each one.
- **Make the agent ask before substituting** a similar colour, the one failure still open.
- **Add unseen test questions** and a human-reviewed set of expected answers, and log retries and tool errors per run.
- **Join sales and recent trend data**, which is where a tool like this becomes useful for design decisions.

## Data note

The images come from a public Hugging Face dataset whose card does not state a licence. The figures here show small product thumbnails for demonstration, and the dataset itself is not included in this repository.
