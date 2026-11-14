---
status: ready
summary: Vietnamese news classification on 2,200 VnExpress articles. LinearSVC reaches 96.06% accuracy on 330 held-out test articles.
problem: Classify Vietnamese news articles into four VnExpress categories (sports, business, science-technology, entertainment) with traditional NLP and machine learning.
dataset: Public VnExpress articles stored in `raw_vnexpress_articles.csv` (`raw_content`, `category`). The corpus contains 2,200 documents after preprocessing. Stratified train/validation/test: 1,540 / 330 / 330.
colab: "https://colab.research.google.com/drive/12hjVvsIH7ZRtqWy4O8hRJd1EycMYoTpL?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/Text_Part_Assignment.ipynb"
pdf: ""
video: ""
---

## Dataset and preprocessing

The dataset consists of 2,200 article bodies collected from VnExpress, with 550 articles from each of four sections. The source section provides the classification label for each article.

Each article body was normalized to Unicode NFC so that Vietnamese characters with diacritics have a consistent representation. The pipeline removed URLs, email addresses, and matching byline or source markers, then lowercased the text, removed punctuation and digits, and collapsed repeated whitespace. PyVi segmented the resulting text into Vietnamese words, joining multi-syllable words such as `công_nghệ` into single tokens.

Only articles with more than 30 tokens were eligible for modeling. Identical cleaned texts within a section were reduced to one copy, and identical texts with conflicting section labels were excluded before splitting. All 2,200 articles met these checks. During feature extraction, TF-IDF excluded the Vietnamese stopwords and represented articles using individual tokens and consecutive token pairs.

## Exploratory analysis

### Article length and writing style

![Article style features by section](assets/text/eda-style-features.png)

The word-count distributions overlap across all four sections. Business articles are concentrated at shorter lengths, while sports has a longer tail. Before lowercasing and removing digits, sports averages **33.63 number sequences** per article versus **19.95** for entertainment. Science and technology has the most uppercase tokens (**15.72** per article), while the boxplots show higher median token lengths for science and business. These differences describe writing style. The classifier uses text after digits and capitalization have been removed.

### Stopword frequency

![Frequent Vietnamese stopwords](assets/text/stopwords.png)

The manual list matches **360,977 of 1,373,286 counted tokens (26.3%)**. “Và” is the most frequent match, with **26,513** occurrences. Filtering common words such as “và”, “của”, and “trong” leaves more of the model’s feature budget for topical vocabulary. The list also removes “không”, so the model may lose some information about negation.

### Frequent terms by section

![Frequent non-stopword terms by section](assets/text/keywords.png)

The counts after stopword filtering surface recognizable topics. Sports uses “bóng” (**2,666**) and “trận” (**2,636**), business uses “đồng” (**1,934**), science and technology uses “ai” (**2,627**) and “công nghệ” (**1,914**), and entertainment uses “phim” (**1,285**). General terms such as “năm” and “người” still rank highly in several sections, so frequency within a section tells only part of the story.

### Common two-token phrases

![Bigrams by section](assets/text/bigrams.png)

Pairs of adjacent segmented tokens add context. “Man Utd” leads the sports ranking (**0.577**), while “tỷ đồng” leads business (**0.623**). “Khoa học công nghệ” and “trí tuệ nhân tạo” appear prominently in science and technology, and “người đẹp” and “cuộc thi” appear in entertainment. The notebook fits this exploratory vectorizer separately to each section’s combined text, so use the scores to rank phrases within each panel.

### Which sections share the most vocabulary?

![Mean pairwise cosine similarity of article TF-IDF vectors by section](assets/text/category-similarity.png)

Each cell averages the cosine similarity between article TF-IDF vectors from the two sections. **Science and business have the highest mean cross-category similarity (0.0601).** Entertainment and sports are next (0.0493). Shared phrases such as “tỷ USD” and “triệu đồng” illustrate the overlap between science and business. The held-out confusion matrix below checks whether the model also confuses those sections. This heatmap uses the full corpus for description. Each classifier fits its own TF-IDF vectorizer using training articles.

## Validation and test results

The stratified 70/15/15 split uses seed 42 and contains 1,540 training, 330 validation, and 330 test articles. Validation and test have 82–83 articles per category.

- **LinearSVC** achieved 0.9757 validation macro F1.
- **Logistic regression** achieved 0.9545 validation macro F1.
- **Multinomial Naive Bayes** achieved 0.9363 validation macro F1.

LinearSVC was selected using validation macro F1. It classified **317 of 330** held-out articles correctly for **96.06% accuracy** and **0.9607 macro F1**.

![Held-out test confusion matrix for LinearSVC](assets/text/confusion-matrices.png)

**Science and business account for 7 of the 13 test errors.** Two examples from the saved notebook output illustrate how their vocabulary can overlap.

- **kinh-doanh → khoa-hoc** — “trong phiên trình bày khai mạc hội nghị y học từ xa châu á ats với chủ đề từ thí điểm đến hạ tầng thể chế hóa…”
- **khoa-hoc → kinh-doanh** — “báo cáo tài chính quý iii công bố tuần này google cho biết công ty hiện có nhân viên trên toàn cầu tính đến cuối…”


## Limitations

**Dataset coverage and labels.** All 2,200 articles come from VnExpress, and the corpus contains an equal number of articles from each section. The reported results therefore reflect a balanced sample from one publisher. Performance may differ on a live news feed with a different mix of topics or on another publisher’s articles. The labels come from editorial sections, so an article can cover several topics while receiving a single label. Science and business accounted for 7 of the 13 test errors, which is consistent with the vocabulary overlap observed in the exploratory analysis.

**Evaluation and text representation.** The results come from one stratified random split. Exact duplicate texts were checked before splitting, but separate articles about the same event can still share names, phrases, or passages across sets. The split also does not measure how well the model handles future news. TF-IDF uses individual tokens and consecutive token pairs, so it has limited ability to connect paraphrases that use different words. Removing digits, capitalization, and stopwords can also discard signals such as scores, financial amounts, acronyms, and negation.
