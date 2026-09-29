---
status: ready
summary: Crawled 2,200 VnExpress articles into four topics; LinearSVC with TF-IDF reaches 98.64% test accuracy.
problem: Classify Vietnamese news articles into four VnExpress topics (sports, business, science-technology, entertainment).
dataset: Self-collected VnExpress corpus — 550 articles per category (2,200 raw; 2,199 after length filtering), 80/20 stratified train/test split (1,759 / 440).
colab: "https://colab.research.google.com/drive/18UUfQYN4BJGZPBsSN3Ajv_YMuP2Us8mB?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/Text_Part_Assignment.ipynb"
pdf: ""
video: ""
---

## EDA

Articles were crawled from four VnExpress sections (`the-thao`, `kinh-doanh`, `khoa-hoc`, `giai-tri`), with videos and podcasts dropped. Preprocessing applies NFC normalization, removes URLs, emails, source/photo notes, punctuation, and digits, then lowercases and segments Vietnamese with PyVi. Articles with 30 tokens or fewer are excluded, leaving 2,199 documents.

### Style features differ by section

![Word count, digits, acronyms, and average word length by news category](assets/text/eda-style-features.png)

Sports copy uses about **13.01** uppercase acronyms per article versus **4.61** in entertainment, and about **34.20** digit tokens versus **17.16** in science.

### Stopwords dominate the raw vocabulary

![Top 20 Vietnamese stopwords and corpus-level stopword share](assets/text/stopwords.png)

Function words make up **28.7%** of **1,276,097** tokens (`và` alone appears 24,632 times), so they are removed before TF-IDF.

### Keywords match the four topics

![Top 15 keywords per news category after stopword removal](assets/text/keywords.png)

Topic tokens separate cleanly: sports (`trận`, `bóng`, `đội`), business (`tăng`, `đồng`, `doanh_nghiệp`), science (`nghiên_cứu`, `khoa_học`), and entertainment (`cô`, `phim`, `diễn_viên`).

### Bigrams recover named entities

![Top 15 TF-IDF bigrams per news category](assets/text/bigrams.png)

TF-IDF bigrams recover entity phrases such as `man utd` / `champions league` in sports and `tỷ đồng` / `nhà đầu_tư` in business, which unigrams miss.

### Categories are lexically far apart

![Pairwise cosine similarity between news categories](assets/text/category-similarity.png)

Mean within-category cosine similarity is **0.1146** for sports, while the closest cross-category pair (`khoa-hoc` vs `kinh-doanh`) is only **0.0405**.

## Training

TF-IDF unigrams and bigrams (`max_features=5000`, Vietnamese stopwords removed) were trained on a stratified 80/20 split. LinearSVC is the strongest of the three sklearn baselines.

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| MultinomialNB | 0.9750 | 0.9751 |
| Logistic Regression | 0.9795 | 0.9796 |
| LinearSVC | 0.9864 | 0.9864 |

![Confusion matrices for MultinomialNB, Logistic Regression, and LinearSVC](assets/text/confusion-matrices.png)

LinearSVC is **1.14 percentage points** above Naive Bayes on 440 test articles and classifies all **110** sports articles correctly. Remaining errors are mostly science vs business.

## Remarks

Self-crawled Vietnamese news is linearly separable with TF-IDF plus a linear classifier. The main limit is a single news source and four coarse labels; next steps are a held-out validation split, more publishers, and error analysis on science–business overlap.
