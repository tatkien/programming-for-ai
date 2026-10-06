---
status: ready
summary: Vietnamese news topic classification on a VnExpress corpus using TF-IDF and three linear sklearn models; LinearSVC reaches 98.64% test accuracy.
problem: Classify Vietnamese news articles into four VnExpress categories (sports, business, science-technology, entertainment) with traditional NLP and machine learning.
dataset: Public VnExpress articles stored in `raw_vnexpress_articles.csv` (`raw_content`, `category`). After length filtering: 2,199 documents; stratified 80/20 train/test (1,759 / 440).
colab: "https://colab.research.google.com/drive/18UUfQYN4BJGZPBsSN3Ajv_YMuP2Us8mB?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/Text_Part_Assignment.ipynb"
pdf: ""
video: ""
---

## Problem and dataset

The task is **single-label Vietnamese news classification**. Each article receives one of four VnExpress section labels: `the-thao`, `kinh-doanh`, `khoa-hoc`, `giai-tri`. The workflow is traditional NLP: clean and segment Vietnamese text, inspect lexical structure, represent documents with TF-IDF, then compare Multinomial Naive Bayes, Logistic Regression, and LinearSVC.

The modeling table is `raw_vnexpress_articles.csv` with `raw_content` (article body) and `category`. A collection cell in the notebook targets **550 articles per section** (2,200 rows in the saved CSV). Length filtering then keeps **2,199** documents. This page reports only those executed counts. Crawl date, request volume, and license-to-redistribute are **not** claimed.

| Field | Role |
| --- | --- |
| `raw_content` | Raw article text used for cleaning and EDA statistics |
| `category` | Section label used as the classification target |
| `clean_content` | Segmented text after `preprocess_vietnamese_text()` |
| `label_id` | Integer encoding of `category` via `LabelEncoder` |

Four-way labels are editorial sections, not fine-grained topics. A science article about markets can still share vocabulary with business, which is the main confusion the models show later.

## Preprocessing

`preprocess_vietnamese_text()` is the modeling text pipeline. Empty or non-string inputs become `''`. Remaining steps are NFC Unicode normalization; removal of URLs, emails, trailing `Ảnh` / `Theo` / `Nguồn` notes, and a leading location prefix (`Hà Nội - ...`); lowercasing; stripping punctuation and **digits**; collapsing whitespace; then **PyVi `ViTokenizer`** word segmentation.

Articles with **30 tokens or fewer** are dropped (`clean_content` word count `> 30`). Labels are encoded with `LabelEncoder`. The executed filter leaves **2,199** rows.

EDA uses a second cleaner, `clean_for_stats()`, so digit counts and uppercase acronyms can be measured **before** those signals are deleted for classification.

**Limitation.** Numbers and capitalization are analyzed, then discarded for TF-IDF. Sports scores, prices, and tokens such as `AI` / `USD` / `HLV` therefore cannot help the classifier.

**Implication.** Linear n-gram models must rely on remaining Vietnamese stems and multiword expressions. That is acceptable if categories are lexically distinct; it is costly if two sections share stems and only differ in numbers or named abbreviations.

## Split

Classification uses a stratified **80/20** split (`test_size=0.2`, `random_state=42`) on `clean_content` with `category` labels: **1,759** train and **440** test documents. A `Pipeline` fits TF-IDF on the training fold only.

There is **no validation set**. Model comparison below is therefore a single test snapshot, not a locked policy chosen on val and then tested once. The notebook previously printed train/test sizes in the preprocess cell without creating that split in the same cell, then split again before fitting. Those two operations are now the same 80/20 split reused through training.

## Exploratory data analysis

EDA is descriptive. Category-level TF-IDF for bigrams, and the cosine matrix, are fit on **all** 2,199 documents. They are not training-only experiments.

### Class counts

The crawl target is balanced: **550** articles in each of the four CSV labels. After the 30-token filter, **one** document is removed (2,200 → 2,199). The test fold used in `classification_report` has **110** articles per class, so accuracy and macro-F1 are close. Overall accuracy is still reported next to macro-F1 so a majority-class reading is not required.

### Style features

![Word-count KDE, mean digits per article, mean uppercase tokens, and average word-length boxplots by category](assets/text/eda-style-features.png)

Four views use `char_count`, `word_count`, `avg_word_length` on `clean_content`, plus digit and uppercase counts on `clean_for_stats()` text.

- Word-count KDEs overlap; section is not a pure length task.
- Mean digit tokens: sports **34.20**, business **27.80**, entertainment **21.50**, science **17.16**.
- Mean uppercase tokens longer than one character: sports **13.01** versus entertainment **4.61**.
- Average characters per word is slightly lower in sports than in business or science.

**Limitation.** These plots include the documents that later enter the test set. Digit and case features are **not** in the classifier.

**Implication.** Sports looks easiest to separate on surface form (acronyms, numbers). Science versus business is the pair to watch: fewer distinctive surface cues after numbers are stripped.

### Stopwords

![Top 20 Vietnamese stopwords and the share of stopword tokens in the corpus](assets/text/stopwords.png)

A **manual** Vietnamese stoplist is applied (including compounds such as `cho_biết`). Over **1,276,097** tokens, stopwords are **28.7%** (**366,408** tokens, **70** distinct stop types in the text). `và` alone occurs **24,632** times.

**Limitation.** A hand list can drop weak content words or keep function words. The list is not tuned per category.

**Implication.** Removing this mass of function tokens is required before TF-IDF; the 5,000-feature cap should then be spent on topical stems.

### Keywords per category

![Top 15 non-stopword unigrams in each VnExpress section](assets/text/keywords.png)

After stopword removal, unigram ranks follow the desks: sports (`trận`, `bóng`, `đội`), business (`tăng`, `đồng`, `doanh_nghiệp`), science (`nghiên_cứu`, `khoa_học`), entertainment (`cô`, `phim`, `diễn_viên`).

**Limitation.** Counts pool every article in a class into one bag. Shared tokens (`đó`, `việt nam` in later n-grams) still overlap.

**Implication.** A unigram+bigram linear model is a fair first classifier because the remaining vocabulary is already section-specific.

### Bigrams

![Top 15 TF-IDF bigrams per category, each class treated as one concatenated document](assets/text/bigrams.png)

Per-class `TfidfVectorizer(ngram_range=(2, 2), max_features=50)` on the concatenated class text surfaces entities (`man utd`, `champions league`, `tỷ đồng`, `nhà đầu_tư`, `nhà khoa_học`).

**Limitation.** This is **not** the classification TF-IDF. Concatenation ignores document frequency inside the class.

**Implication.** Classification should include bigrams (`ngram_range=(1, 2)`), which it does, so entity phrases can fire at prediction time.

### Cosine similarity between categories

![Mean pairwise cosine similarity of TF-IDF document vectors between categories](assets/text/category-similarity.png)

A global `TfidfVectorizer(max_features=5000)` is fit on all `clean_content`. For each category pair the mean of pairwise document cosines is stored. Within-class similarity is highest for sports (**0.1146**). The closest cross-class pair is science–business (**0.0405**); science–sports is **0.0274**.

Mean cosine similarity of TF-IDF vectors $A$ and $B$ is

$$\cos(\theta)=\frac{A\cdot B}{\|A\|\|B\|}$$

**Limitation.** The vectorizer sees the full corpus, including test documents. Absolute cosine values are small because news articles are sparse; ranking of pairs matters more than the raw scale.

**Implication.** Categories are lexically far apart on average. Residual errors should concentrate on science vs business, matching the confusion matrices.

## Modeling

All three models share one sklearn pipeline: TF-IDF features, then a classifier. The vectorizer keeps at most 5,000 unigram and bigram features and drops the manual Vietnamese stopwords. It is fitted on the training fold only.

Multinomial Naive Bayes is the probabilistic baseline for this sparse feature space. Logistic Regression is a linear baseline on the same features, with at most 1,000 solver iterations and regularization C equal to 1. LinearSVC is a linear support vector machine on the same matrix, with C equal to 1 and no grid search.

No PhoBERT, BERT, or other pretrained Vietnamese encoder is trained. The hyperparameters above are fixed; there is no cross-validation search.

## Evaluation

Metrics come from `classification_report` on **440** test articles (**110** per class): precision, recall, F1, and support, plus accuracy.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | --- | --- | --- | --- |
| MultinomialNB | 0.9750 | 0.9752 | 0.9750 | 0.9751 |
| Logistic Regression | 0.9795 | 0.9799 | 0.9795 | 0.9796 |
| LinearSVC | 0.9864 | 0.9864 | 0.9864 | 0.9864 |

LinearSVC per-class F1 on the same fold: entertainment **0.9863**, science **0.9820**, business **0.9772**, sports **1.0000**.

![Confusion matrices for MultinomialNB, Logistic Regression, and LinearSVC on the 440-article test set](assets/text/confusion-matrices.png)

Rows are true labels; columns are predictions. For LinearSVC, sports is **110 / 110**. Off-diagonal mass is small and sits on **science–business** (and a few entertainment leaks). Naive Bayes is **1.14 percentage points** below LinearSVC in accuracy.

**Limitation.** The notebook outputs do not include the raw text of each misclassified article. The confusion matrix is category-level only. A follow-up cell prints true/predicted labels and a token snippet for LinearSVC errors when the notebook is re-run.

**Implication.** High accuracy is consistent with the cosine and keyword plots: three desks are easy; the remaining mistakes are the semantically closer pair the EDA already flagged.

## Limitations

- **Single publisher.** All text is VnExpress style and VnExpress section names. Results do not transfer to other Vietnamese outlets without new data.
- **Academic scrape, not a licensed dump.** Articles were taken from publicly accessible pages for this course. The notebook is not evidence of a redistribution license, so the CSV is not published in the repository.
- **Manual stoplist.** The list may be incomplete or too aggressive.
- **Numbers and case dropped** in `preprocess_vietnamese_text()` while EDA still uses them.
- **TF-IDF is lexical.** Paraphrases with little n-gram overlap look dissimilar.
- **Traditional models only.** No Transformer ablation, so this is not a SOTA Vietnamese NLP comparison.
- **No validation split and no tuning.** `C=1.0` and default NB are frozen. The test set is the only reported holdout.
- **Descriptive TF-IDF vs model TF-IDF.** Class-level bigrams and the cosine matrix use the full corpus; classification TF-IDF is inside the train-only pipeline.
- **No saved article-level error table** in the current executed outputs.

## Conclusion

EDA shows a **balanced four-way section task** whose remaining vocabulary, after Vietnamese segmentation and stopword removal, is already desk-specific. Mean cross-class cosine similarity stays low (**0.0405** at most for science–business). In that setting, unigram+bigram TF-IDF plus a linear classifier is the right first model: LinearSVC reaches **0.9864** accuracy and macro-F1 on 440 test articles, **1.14 points** above Naive Bayes, with perfect sports recall. The same science–business overlap that appears in EDA is the residual confusion. Limits are the VnExpress-only source, features that throw away numbers and case, a single un-tuned test split, and no inspection of individual misclassified pieces in the stored outputs.
