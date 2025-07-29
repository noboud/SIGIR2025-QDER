# QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking

📄 Shubham Chatterjee and Jeff Dalton. 2025. [DREQ: Document Re-Ranking Using Entity-based Query Understanding](https://dl.acm.org/doi/10.1145/3726302.3730065). In _Proceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval (SIGIR ’25)._ 

This repository contains the code associated with this paper and the instructions on how to execute the code. For detailed instructions on how to run the code, read the documentation (**_detailed documentation with data release coming soon!_**). 

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

All data associated with this work is licensed and released under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


## Acknowledgement
This material is based upon work supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/V025708/1. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the EPSRC.

---

## 🧪 Running the code

The code was tested using python v3.9.21.

### ✅ Step 1: Environment Setup

```bash
git clone https://github.com/shubham526/SIGIR2025-QDER.git
cd SIGIR2025-QDER
python -m venv qder_env
source qder_env/bin/activate  
pip install -r requirements.txt
```

---

### 📦 Step 2: Data Preparation

#### Required Files:

* `queries.tsv`: query ID + text
* `corpus.jsonl`: docs with linked entities
* `bm25.run`: initial document rankings
* `qrels.txt`: relevance labels
* `entity_embeddings.jsonl`: entity ID + embedding + score

#### Run the Pipeline:

```bash
# Generate document-entity run
python scripts/data_preparation/create_entity_run.py \
    --docs data/corpus.jsonl \
    --run data/bm25.run \
    --save data/entity_run.run

# Create query-document pairs for reranking
python scripts/data_preparation/create_rerank_data.py \
    --queries data/queries.tsv \
    --docs data/corpus.jsonl \
    --qrels data/qrels.txt \
    --doc-run data/bm25.run \
    --entity-run data/entity_run.run \
    --embeddings data/entity_embeddings.jsonl \
    --save data/train.jsonl \
    --train --balance
```

Repeat for `dev.jsonl` and `test.jsonl`.

---

### 🏋️ Step 3: Train the QDER Model

```bash
python scripts/train.py \
    --train-data data/train.jsonl \
    --dev-data data/dev.jsonl \
    --qrels data/qrels.txt \
    --text-enc bert-base-uncased \
    --use-entities \
    --use-scores \
    --score-method bilinear \
    --epochs 20 \
    --batch-size 8 \
    --output-dir experiments/qder_baseline \
    --save-best
```

---

### 🧪 Step 4: Evaluate the Model

```bash
python scripts/test.py \
    --checkpoint experiments/qder_baseline/best_model.pt \
    --test-data data/test.jsonl \
    --qrels data/test_qrels.txt \
    --save-run results/qder_baseline.run \
    --metric map
```

---

### 🧬 Step 5: Run Ablation Studies

```bash
python scripts/ablation_study.py \
    --train-data data/train.jsonl \
    --val-data data/dev.jsonl \
    --qrels data/qrels.txt \
    --pretrained-model bert-base-uncased \
    --output-dir experiments/ablation \
    --variants no_add no_subtract no_multiply no_interactions
```


## 📚 Citation

```bibtex
@inproceedings{10.1145/3726302.3730065,
author = {Chatterjee, Shubham and Dalton, Jeff},
title = {QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking},
year = {2025},
isbn = {9798400715921},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3726302.3730065},
doi = {10.1145/3726302.3730065},
abstract = {Neural IR has advanced through two distinct paths: entity-oriented approaches leveraging knowledge graphs and multi-vector models capturing fine-grained semantics. We introduce QDER, a neural re-ranking model that unifies these approaches by integrating knowledge graph semantics into a multi-vector model. QDER's key innovation lies in its modeling of query-document relationships: rather than computing similarity scores on aggregated embeddings, we maintain individual token and entity representations throughout the ranking process, performing aggregation only at the final scoring stage-an approach we call ''late aggregation.'' We first transform these fine-grained representations through learned attention patterns, then apply carefully chosen mathematical operations for precise matches. Experiments across five standard benchmarks show that QDER achieves significant performance gains, with improvements of 36\% in nDCG@20 over the strongest baseline on TREC Robust 2004 and similar improvements on other datasets. QDER particularly excels on difficult queries, achieving an nDCG@20 of 0.70 where traditional approaches fail completely (nDCG@20 = 0.0), setting a foundation for future work in entity-aware retrieval.},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2255–2265},
numpages = {11},
keywords = {multi-vector entity-oriented search, query-specific embedding},
location = {Padua, Italy},
series = {SIGIR '25}
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repo
2. Create a feature branch
3. Submit a PR with a detailed message

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.

---

## 🙋 Contact

* For questions about the code or paper, contact Shubham Chatterjee at shubham.chatterjee@mst.edu

---
