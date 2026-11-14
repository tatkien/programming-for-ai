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

## Preprocessing examples

One article from each VnExpress section shows how preprocessing changes the text. Each excerpt shows up to 50 whitespace-separated units from the original or 50 tokens after cleaning and PyVi segmentation. Underscores join syllables that PyVi treats as one word.

### Entertainment (`giai-tri`)

**Before preprocessing** · Ngày 26/10, Harper's Bazaar có bài phỏng vấn Natalie Portman dịp diễn viên sắp trở lại màn ảnh với The Gallerist (dự kiến ra mắt hôm 4/12). Ngoài thảo luận về phim mới, cô nói về cuộc sống khi có con thứ ba. Em bé là con của Portman và

**After preprocessing** · ngày harper s bazaar có bài phỏng_vấn natalie portman dịp diễn_viên sắp trở_lại màn_ảnh với the gallerist dự_kiến ra_mắt hôm ngoài thảo_luận về phim mới cô nói về cuộc_sống khi có con thứ ba em bé là con của portman và nhạc_sĩ người pháp tanguy destable con đầu_lòng của anh

### Science and technology (`khoa-hoc`)

**Before preprocessing** · Theo NBC News , từ tháng 6 đến nay, kẻ trộm đã lấy đi lượng hàng hóa trị giá hơn 100 triệu USD, vốn đang được vận chuyển đến các trung tâm dữ liệu Mỹ. Con số này được ước tính dựa trên thông tin từ lãnh đạo doanh nghiệp,

**After preprocessing** · theo nbc news từ tháng đến nay kẻ_trộm đã lấy đi lượng hàng_hóa trị_giá hơn triệu usd vốn đang được vận_chuyển đến các trung_tâm dữ_liệu mỹ con_số này được ước_tính dựa trên thông_tin từ lãnh_đạo doanh_nghiệp cố_vấn và điều_tra_viên hỗ_trợ cơ_quan cảnh_sát điều_tra những vụ trộm lớn nhất hàng_hóa bị

### Business (`kinh-doanh`)

**Before preprocessing** · Sau phiên giảm sâu đầu tuần, nhiều nhóm phân tích khuyến nghị nhà đầu tư hạ tỷ trọng cổ phiếu, bởi thị trường đang thiếu lực cầu nâng đỡ và có khả năng thủng thêm các ngưỡng hỗ trợ quan trọng. Thực tế hôm nay VN-Index giao dịch dưới tham

**After preprocessing** · sau phiên_giảm sâu đầu tuần nhiều nhóm phân_tích khuyến_nghị nhà đầu_tư hạ tỷ_trọng cổ_phiếu bởi thị_trường đang thiếu lực cầu nâng_đỡ và có khả_năng thủng thêm các ngưỡng hỗ_trợ quan_trọng thực_tế hôm_nay vn index giao_dịch dưới tham_chiếu trong phần_lớn thời_gian và nới rộng biên_độ giảm ở những phút cuối chỉ_số

### Sports (`the-thao`)

**Before preprocessing** · Sự việc xảy ra tại sân Parque do Sabia, thành phố Uberlandia, bang Minas Gerais, Brazil, hôm 1/11. Trận lượt đi bán kết giải hạng Nhì bang Minas Gerais dành cho lứa U20 kết thúc với chiến thắng 3-1 cho Mamore, nhưng bàn thắng thứ ba của đội khách châm

**After preprocessing** · sự_việc xảy ra tại sân parque do sabia thành_phố uberlandia bang minas gerais brazil hôm trận lượt đi bán_kết giải hạng nhì bang minas gerais dành cho lứa u kết_thúc với chiến_thắng cho mamore nhưng bàn thắng thứ ba của đội khách châm ngòi cho hỗn_chiến theo biên_bản của trọng_tài

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
