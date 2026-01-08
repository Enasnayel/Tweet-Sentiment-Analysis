# Sentiment Tweet Analysis (Positive vs Negative)

## Project Overview
This project builds an end-to-end NLP pipeline to classify tweet sentiment into **negative** and **positive**. Tweets are short and noisy (slang, abbreviations, informal language), so automated sentiment analysis helps support scalable and timely interpretation of public opinion.

## Dataset
- **Source:** Hugging Face  
- **Dataset:** `gxb912/large-twitter-tweets-sentiment`  
- **Original Size:** 224,994 rows (tweet text + sentiment label)  
- **Final Working Size (after cleaning/filtering):** [e.g., 214,282 rows]  
- **Target:** Binary sentiment classification (**negative vs positive**)

## Workflow (Rubric Tasks)
### Task 1 — Problem Definition & Motivation
Organizations cannot manually analyze large volumes of tweets reliably and quickly due to high volume and noisy text. This project addresses that by applying NLP + machine learning to automate sentiment classification.

### Task 2 — Dataset Selection & Justification
The dataset was selected because it is tweet-based, intended for sentiment classification, large enough for effective training, and suitable for binary sentiment modeling.

### Task 3 — Text Preprocessing (Detailed)
Tweet text contains noise that can reduce model accuracy. The following preprocessing steps were applied in sequence to create a clean text column (`ctext`) and token-based features (`ftokens`).

## Additional Preprocessing Resource (Slang Dictionary)

To expand chat-style slang and abbreviations commonly found in tweets, I used an external slang/acronym dictionary from Kaggle:

- **Dataset:** Chat Slang Abbreviations Acronyms  
- **Source:** Kaggle (gowrishankarp)  
- **File used:** `slang.csv` (columns: `acronym`, `expansion`)  
- **How it was used:** Converted into a lookup dictionary (`slang_dict`) and applied during cleaning to replace acronyms with their full expansions before tokenization and feature extraction.

#### Cleaning & Normalization Pipeline (ctext)
1. **Remove URLs**  
   - Purpose: links do not contribute to sentiment and add noise.  
   - Method: regex removal of patterns like `http...` and `www...`

2. **Remove user mentions (`@username`)**  
   - Purpose: usernames are not useful features and add sparsity.  
   - Method: regex removal of `@\w+`

3. **Expand slang/acronyms using a dictionary (`slang_dict`)**  
   - Purpose: convert informal acronyms to meaningful words for better feature extraction.  
   - Example: `idk → i do not know` (depends on dictionary content)

4. **Normalize contractions**  
   - Purpose: reduce variation and improve token consistency.  
   - Examples:  
     - `can't → can not`  
     - `I'm → I am`  
     - `we're → we are`

5. **Normalize Twitter shorthand**  
   - Purpose: convert common shorthand to standard words.  
   - Examples:  
     - `2day → today`  
     - `2moro / 2mrw → tomorrow`  
     - `u → you`

6. **Normalize elongated words**  
   - Purpose: reduce repeated letters used for emphasis and make tokens consistent.  
   - Example: `soooo → soo`

7. **Normalize emoticons into sentiment words** *(dataset contains emoticons ~1.47%)*  
   - Purpose: preserve sentiment signals that would be lost during punctuation removal.  
   - Examples:  
     - `:) / :-) → smile`  
     - `:( / :-( → sad`  
     - `:D / XD → laugh`  
     - `<3 → love`

8. **Normalize colon-based emotion markers** *(e.g., `:giggles:` / `::cries::`)*  
   - Purpose: keep emotion words as tokens rather than removing them as punctuation.  
   - Examples:  
     - `:giggles: → giggles`  
     - `::cries:: → cries`

9. **Remove digits**  
   - Purpose: reduce noisy numeric tokens that typically don’t help sentiment classification.

10. **Clean hashtags**  
   - Purpose: remove the `#` symbol while keeping the keyword.  
   - Example: `#happy → happy`

11. **Remove punctuation**  
   - Purpose: reduce token fragmentation and normalize text.

12. **Normalize whitespace**  
   - Purpose: remove extra spaces and ensure clean token boundaries.

#### Tokenization & Feature Columns
- **Tokenization:** NLTK `TweetTokenizer`  
  - Settings used: lowercase conversion, handle stripping, and repeated character normalization.
- **Lemmatization:** NLTK `WordNetLemmatizer`  
  - Purpose: reduce words to base form (e.g., “cars” → “car”) to reduce sparsity.
- Output columns created:
  - `ctext` = cleaned text  
  - `tokens` / `ftokens` = token lists used for downstream representation and modeling  
  - `ftext` = final cleaned text string for vectorizers

### Task 4 — Text Representation
- **Bag of Words + N-grams (Unigram + Bigram)**
- **TF-IDF (Unigram + Bigram)**
- **POS Tagging**  
- **Named Entity Recognition (NER)** *(if implemented)*

### Task 5 — Machine Learning Implementation & Evaluation
Models implemented:
- **TF-IDF + Logistic Regression** *(final selected model)*
- **TF-IDF + Linear SVM**
- Transformer models: **BERT, RoBERTa, BERTweet** *(used for comparison)*

Evaluation metrics:
- Classification Report: Accuracy, Precision, Recall, F1-score  
- Confusion Matrix  
- ROC Curve + AUC (for models that provide probability/decision scores)

### Task 6 — Storytelling & Insights
Visualizations were used to support:
- token frequency patterns  
- n-gram importance  
- model comparison using performance metrics and confusion matrices  
- ROC/AUC for ranking performance comparison (where applicable)

## Results (Main)

- **TF-IDF + Logistic Regression:** Accuracy ≈ 0.76, Macro-F1 ≈ 0.75, ROC-AUC ≈ 0.837  
- **TF-IDF + Linear SVM:** Accuracy ≈ 0.74, Macro-F1 ≈ 0.74  

Logistic Regression was selected due to slightly better balanced performance and availability of probability outputs for ROC/AUC and threshold tuning.

