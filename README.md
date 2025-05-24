# BiLegalNERD
A Bilingual Legal NER Dataset and Semantics-Aware Dual Translation for Low-Resource Languages

This repository contains the dataset construction tools and model training code for our research on **Named Entity Recognition (NER) in low-resource legal domains**, with a focus on cross-lingual transfer from Chinese to Uyghur.

---

## 🔍 Project Overview

This work addresses the challenge of legal NER in low-resource languages by introducing a bilingual Uyghur–Chinese NER dataset and a semantics-aware annotation transfer method. The proposed framework includes:

- **BiLegalNERD**: A bilingual legal NER dataset constructed from Chinese legal texts and aligned Uyghur translations, annotated with ten fine-grained legal entity types.
- **CUTLM**: A semantics-aware data transfer approach based on **dual translation** and **Levenshtein distance alignment**, designed to preserve entity boundaries during translation.
- **BiLegalNER**: A domain-adapted NER model enhanced by **vocabulary expansion** and **bilingual fine-tuning**.

---

## 📁 Repository Structure

```bash
BiLegalNERD/
├── README.md         # Project introduction and usage instructions
├── cutlm.py          # Dual translation and Levenshtein-based alignment (CUTLM method)
├── trainer.py        # NER model training pipeline
├── ugnerd.py         # Inference and evaluation on Uyghur legal texts

🚀 Usage
1. Annotation Transfer (CUTLM)
python cutlm.py --input data/chinese_annotated.json --output data/uyghur_aligned.json

2. Model Training
python trainer.py --train_file data/uyghur_aligned.json --model cino --output_dir ./models/bilegalner

3. Inference and Evaluation
python ugnerd.py --model_dir ./models/bilegalner --test_file data/uyghur_test_manual.json

📊 Entity Types
The dataset includes 10 fine-grained legal NER tags such as:

NHCS: Suspect names
NHVI: Victim names
NS: Locations
NT: Time expressions
NO: Organizations
NCGV / NCSM / NASI / NCSP / NATS: Various legal-specific case elements

🔗 Notes
All dataset files have been preprocessed to support BIO-format tagging.
The CUTLM method is agnostic to translation APIs but was tested using Google Translate and iFlytek for cross-validation.
For research use only. Please cite appropriately if used in derived work (citation info to be updated upon formal publication).
