# Aspects Recognition in English Texts

## Definition

Aspect extraction in NLP involves identifying and extracting specific aspects, attributes, or features mentioned in text data. 
This task is crucial for understanding the finer details and nuances of textual information, especially in domains like product reviews, 
customer feedback, social media analysis, and opinion mining. 

### Summary
This project focuses on solving the problem of Aspect Recognition in English text. There are three solutions available in the repository:

1. **Solution 1: Frequency-Based Method**
   - Initial text processing involved tokenization, removing stopwords, and applying stemming using the PorterStemmer() function from nltk.
   - Part-of-Speech tagging (Pos_Tag()) was applied to sentences, followed by extracting nouns and noun phrases using the noun_chuncks() function from spacy.
   - Aspects were identified based on their frequency, retaining those with frequencies above a specified threshold.

2. **Solution 2: H&L Method**
   - Inspired by [Hu & Liu's](https://cdn.aaai.org/AAAI/2004/AAAI04-119.pdf) approach, this method addresses issues like repeated nouns potentially becoming aspects.
   - It first extracts nouns and noun phrases, considers them as candidate aspects, and then forms new aspect candidates by concatenating pairs and trios of aspects occurring in the same order within a sentence.
   - Aspect candidates are ranked based on their p-support, which is the number of sentences containing the aspect excluding sentences containing another aspect.
   - Further pruning is done by eliminating non-compact phrases and candidates with low p-support or included within another candidate.

3. **Solution 3: Supervised Learning with Conditional Random Field (CRF)**
   - As aspect extraction can be framed as a sequence labeling problem, Conditional Random Field (CRF) was applied.
   - Features used included part-of-speech tags, lemmas, word frequencies, and semantic context (neighboring words).
   - Evaluation was performed using precision, recall, and F1-score metrics, visualized through the LETO visual interface.

### Additional Notes
- The state-of-the-art includes syntax-based methods focusing on syntactic relationships, and unsupervised learning using Latent Dirichlet Allocation (LDA).
- Syntax-based methods excel in capturing less frequent aspects but require numerous rules, while LDA-based approaches offer an unsupervised alternative.
