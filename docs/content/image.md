---
status: draft
summary: EDA-first study of plant-disease localization and classification using a classical proposal pipeline and YOLO26n
problem: Detect plant regions in field photographs and classify 27 disease and health classes
dataset: "[Roboflow FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant), 5,156 images and 8,580 annotated boxes. Paper: [doi:10.1109/ACCESS.2023.3263042](https://doi.org/10.1109/ACCESS.2023.3263042)."
colab: "https://colab.research.google.com/drive/1iIgivdhGo6RJguEbPvl4Yfu96qKUg7ZO?usp=sharing"
notebook: "https://github.com/tatkien/programming-for-ai/blob/main/image_part_assignment.ipynb"
pdf: ""
video: ""
---

## Problem and dataset

FieldPlant contains **5,156 field photographs, 8,580 annotated objects, and 27 classes** from cassava, corn, and tomato. The task has two parts: locate each annotated plant region, then assign its class. The images include clean single-object cases, crowded plants, large changes in object size, and background vegetation that resembles the target.

Original source: [Roboflow FieldPlant](https://universe.roboflow.com/plant-disease-detection/fieldplant). Paper: [doi:10.1109/ACCESS.2023.3263042](https://doi.org/10.1109/ACCESS.2023.3263042).

### Tomato examples

![Tomato images showing one annotated leaf, several leaves, the smallest tomato box, and the largest tomato box](assets/image/tomato_samples.png)

The examples show why a single summary image is not representative. Some photographs contain one clear leaf, while others contain several overlapping annotations. Box size also varies from small local regions to an object covering most of the photograph.

## Exploratory data analysis

### Class imbalance and split coverage

![Horizontal bars comparing distinct training images and training boxes for all 27 classes](assets/image/class_distribution.png)

The three largest classes, Tomato Brown Spots, Cassava Mosaic, and Corn leaf blight, account for **64.2% of all training boxes**. **Ten classes have fewer than 20 training boxes.** At the extreme, Corn Charcoal has one training image and one box, while Tomato bacterial wilt has one training image and two boxes. **Neither class has a validation or test example**, so held-out performance cannot be estimated for them.

**Overall accuracy would hide weak performance on rare classes** because common classes dominate it. We therefore report macro-F1 and per-class support, and use class-aware repetition when training YOLO. Classes with 20 to 99 training boxes receive 2x repetition, classes with 10 to 19 receive 3x, and classes with fewer than 10 receive 4x.

### Objects per image

![Histogram of valid annotated boxes per training image](assets/image/leaf_counts.png)

Among 3,609 readable training images, **63.3% contain one annotated object, 36.7% contain multiple objects, and 17.5% contain more than two**. The classical proposal stage keeps at most two boxes per image. Even if those boxes were placed perfectly, **this rule limits count-based recall to 82.3%** because additional objects cannot be returned.

### Where annotations appear

![Heatmaps of normalized annotation centers for all plants and for cassava, corn, and tomato](assets/image/box_center_heatmap.png)

Across the training set, **65.6% of annotation centers fall inside the central quarter of the image**. Cassava and corn show a particularly clear center concentration, while tomato is more dispersed. This describes how the dataset was photographed and annotated; it does not mean that real objects always occur near the center. Error analysis must still include objects near image edges.

### Classes that occur together

![Conditional co-occurrence heatmap for class pairs found in the same training image](assets/image/class_cooccurrence.png)

**Seven distinct class pairs occur in the same training images.** The strongest supported pair is Tomato healthy with Tomato blight leaf: **93 images contain both**. Because the heatmap is conditional, direction matters. Those 93 images represent **89.4% of images containing Tomato healthy but only 37.3% of images containing Tomato blight leaf**.

Some large percentages come from only one or a few images and should not be treated as stable relationships. The split keeps all boxes from an image file together, preventing annotations from the same file from entering different splits. A separate duplicate audit found identical files across splits; that issue is reported under limitations.

### HSV and HOG feature dimensionality

![Cumulative explained variance from PCA on class-capped HSV and HOG training features](assets/image/pca_cumulative_variance.png)

The first two principal components retain only **21.2% of the variance** in the HSV and HOG representation. Reaching 80% requires **132 components**, and reaching 90% requires **227 components**. The handcrafted representation is therefore spread across many dimensions. A two-dimensional PCA projection is useful for inspection, but overlap in that projection cannot establish that two diseases are inseparable.

### ExG and HSV proposal behavior

Excess Green, or ExG, emphasizes pixels whose green channel is stronger than red and blue. HSV thresholds then restrict hue, saturation, and brightness to suppress unlikely vegetation pixels. Connected mask regions become candidate boxes.

![Low-coverage ExG and HSV example with fragmented segmentation and no matched proposal](assets/image/exg_hsv_low_coverage.png)

In the low-coverage example, discoloration and background vegetation fragment the mask. The two retained blue proposals cover small regions rather than the large red ground-truth object, so no proposal reaches IoU 0.5.

![High-coverage ExG and HSV example with a proposal matching the ground-truth plant region](assets/image/exg_hsv_high_coverage.png)

The high-coverage example has a clean green foreground and produces one proposal that closely follows the ground-truth box. On 100 sampled training images, however, **the top-two ExG and HSV procedure covers only 29.4% of 153 ground-truth boxes**, with **mean best IoU 0.375**. Clean examples therefore do not represent its typical coverage.

## Modeling choices

### YOLO augmentation and class-aware repetition

![Original crops and YOLO-style augmented previews for the one-times through four-times sampling tiers](assets/image/yolo_augmentation_preview.png)

YOLO training uses horizontal and vertical flips, rotation up to 15 degrees, translation, scaling, mild HSV and contrast changes, Mosaic, and limited MixUp. Underrepresented classes are also repeated according to the training-box tiers described above. Validation and test images remain unchanged.

The classical baseline trains on raw crops. Keeping it unaugmented provides a clear handcrafted-feature reference and avoids changing both the feature representation and training distribution at the same time.

### Classical proposal methods

The classical pipeline evaluates three proposal mechanisms with a strict green mask and an expanded green-yellow-brown mask.

- **ExG and HSV:** form connected regions directly from vegetation-colored pixels.
- **Watershed:** uses distance-transform markers to separate connected foreground regions before boxes are fitted.
- **GrabCut:** starts from a coarse rectangle and iteratively separates likely foreground from background using color models and graph cuts.

| Proposal method | Mean best IoU | Hit rate at IoU 0.5 | Mean proposals per image |
| --- | --- | --- | --- |
| Watershed, green mask | 0.364 | 32.9% | 1.94 |
| Watershed, expanded mask | 0.365 | 31.9% | 1.86 |
| ExG and HSV, green mask | 0.339 | 27.1% | 1.90 |
| ExG and HSV, expanded mask | 0.316 | 24.4% | 1.73 |
| GrabCut, green mask | 0.327 | 26.3% | 1.25 |
| GrabCut, expanded mask | 0.327 | 26.3% | 1.25 |

![Validation examples comparing red ground-truth boxes with blue boxes from the selected classical proposal method](assets/image/proposal_methods.png)

**The expanded masks do not consistently improve matching.** Field clutter, overlapping leaves, shadows, severe discoloration, and annotations that include ears or tassels can merge objects, fragment one object into several regions, or select background vegetation. Watershed with the green mask has the highest validation hit rate, but **it still localizes only 32.9% of the objects**.

### How proposal errors affect classification

Using the correct ground-truth crops, the selected enhanced HGB classifier reaches **validation macro-F1 0.432**. When the same classification stage receives matched watershed proposals, **matched-proposal macro-F1 falls to 0.348**. Imperfect boxes include irrelevant background or omit disease regions, changing the features presented to the classifier.

The larger failure occurs before classification: **only 434 of 1,318 validation objects receive a proposal with IoU at least 0.5**. The remaining **884 objects are localization false negatives**, so the classifier never sees them. The low-coverage ExG and HSV example above illustrates this mechanism without claiming a specific class prediction for an unmatched crop.

## YOLO26n results

![Validation examples comparing human ground-truth boxes in red with YOLO26n predictions in blue](assets/image/yolo_predictions.png)

YOLO jointly learns localization and class prediction. In these examples, blue prediction boxes closely follow the red annotations. Numbers printed beside YOLO labels are confidence scores; the separate matching analysis counts a localization hit when IoU is at least 0.5.

Standard YOLO detection metrics:

| Metric | Validation | Test |
| --- | --- | --- |
| mAP at 0.5 | 0.695 | 0.744 |
| mAP at 0.5 to 0.95 | 0.548 | 0.609 |
| Precision | 0.686 | 0.676 |
| Recall | 0.689 | 0.675 |

Classical and YOLO parity on the validation set:

| Method | Matched macro-F1 | Hit rate at IoU 0.5 |
| --- | --- | --- |
| Watershed green plus enhanced HGB | 0.348 | 32.9% |
| YOLO26n with augmentation and oversampling | 0.802 | 84.4% |

**The main difference is localization coverage. YOLO matches 1,113 of 1,318 validation objects, compared with 434 for the classical pipeline.** Among matched objects, **YOLO raises macro-F1 from 0.348 to 0.802**. Validation and test mAP describe different image sets, so the higher test value should not be interpreted as improvement caused by testing.

### Augmentation ablation from matched control runs

![Per-class validation recall changes for augmentation and augmentation plus oversampling relative to the YOLO control](assets/image/yolo_recall_change.png)

Positive bars show recall gains over the control run and negative bars show losses. **Augmentation plus oversampling improves several underrepresented classes, but it does not improve every class.** Gains are visible for Corn Purple Discoloration, Corn Insects Damages, and Cassava Bacterial Blight. Changes for classes with only one to four validation objects are highly sensitive to a single detection and should be read together with class support.

## Limitations

- **Cross-split duplicates:** the audit finds 113 groups of byte-identical files spanning different splits. This leakage can make validation and test results optimistic.
- **Missing holdout classes:** Corn Charcoal and Tomato bacterial wilt are absent from validation and test, so the study cannot report held-out performance for them.
- **Small rare-class samples:** several rare classes have only one to four validation examples. Perfect or zero recall on such classes is not a stable estimate.
- **Matched-only macro-F1:** this metric uses only predictions paired with ground truth at IoU at least 0.5. It excludes unmatched ground-truth objects and unmatched predictions, which are reported separately through hit rate and error counts.
- The control, augmentation-only, and augmentation-plus-oversampling YOLO experiments came from matched but separate runs. Their comparison depends on preserving the same split, seed, training settings, and evaluation procedure.
- All images come from the same source dataset. Performance on other farms, cameras, lighting conditions, and regions remains unknown.

## Conclusion

The EDA identifies two central difficulties: **severe class imbalance** and **unreliable color-based localization in crowded field scenes**. The enhanced classical classifier performs reasonably on correct crops, but the top-two watershed proposal stage finds only **32.9% of validation objects**. **YOLO26n raises localization coverage to 84.4% and matched macro-F1 to 0.802**, making localization the clearest source of its advantage. Duplicate leakage and very small rare-class holdouts remain the main constraints on these conclusions.
