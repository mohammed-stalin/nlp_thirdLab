# Language-Modeling

## Overview
This project focuses on utilizing Natural Language Processing (NLP) techniques and models provided by the Sklearn library. The objective is to gain familiarity with language models for regression and classification tasks.

## Part 1: Language Modeling / Regression

### Dataset
The dataset used for this part can be found here: [answers.csv](https://github.com/dbbrandt/short_answer_granding_capstone_project/blob/master/data/sag/answers.csv).

### Preprocessing
- Tokenization, stemming, lemmatization, and removal of stop words are performed to preprocess the collected dataset.
- Discretization is applied to transform continuous data into categorical data.

### Encoding
Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF are used to encode the data vectors.

### Models
The following models are trained using the Word2Vec embeddings:
- Support Vector Regression (SVR)
- Linear Regression
- Decision Tree

### Evaluation
The models are evaluated using standard metrics such as Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The best model is selected based on these metrics, and the choice is justified.

### Results Interpretation
The results from the evaluation of different models and vectorization techniques are as follows:

1. **SVR with TF-IDF**
   - **MSE:** 0.9578
   - **RMSE:** 0.9787

2. **Linear Regression with TF-IDF**
   - **MSE:** 2.6744
   - **RMSE:** 1.6354

3. **Decision Tree with TF-IDF**
   - **MSE:** 1.6899
   - **RMSE:** 1.3000

4. **SVR with CBOW**
   - **MSE:** 1.7285
   - **RMSE:** 1.3147

5. **Linear Regression with CBOW**
   - **MSE:** 1.2975
   - **RMSE:** 1.1391

6. **Decision Tree with CBOW**
   - **MSE:** 1.8476
   - **RMSE:** 1.3593

7. **SVR with Skip Gram**
   - **MSE:** 1.4878
   - **RMSE:** 1.2198

8. **Linear Regression with Skip Gram**
   - **MSE:** 1.2330
   - **RMSE:** 1.1104

9. **Decision Tree with Skip Gram**
   - **MSE:** 1.7355
   - **RMSE:** 1.3174

10. **SVR with Bag of Words**
    - **MSE:** 1.1196
    - **RMSE:** 1.0581

11. **Linear Regression with Bag of Words**
    - **MSE:** 5.1984
    - **RMSE:** 2.2800

12. **Decision Tree with Bag of Words**
    - **MSE:** 1.6574
    - **RMSE:** 1.2874

### Analysis and Conclusion
1. **Best Performing Model and Vectorization Method:**
   - **SVR with TF-IDF** has the lowest MSE (0.9578) and RMSE (0.9787). This indicates that this combination has the best performance in terms of minimizing the error in the predictions.
   - **SVR with Bag of Words** also performs well with an MSE of 1.1196 and an RMSE of 1.0581, showing it’s a strong alternative.

2. **Linear Regression Performance:**
   - Linear Regression with TF-IDF, CBOW, and Skip Gram show moderate performance, but performs poorly with Bag of Words.
   - The highest MSE and RMSE were observed with Linear Regression using Bag of Words, making it the least effective model and vectorization combination.

3. **Decision Tree Performance:**
   - Decision Tree with TF-IDF has a reasonably low MSE and RMSE, making it a decent choice.
   - However, Decision Tree generally does not outperform SVR models.

4. **Effectiveness of Vectorization Techniques:**
   - **TF-IDF** consistently showed strong performance, especially when combined with SVR, highlighting its effectiveness in capturing the importance of words in the documents.
   - **Bag of Words** also showed strong performance with SVR but was less effective with Linear Regression.
   - **CBOW and Skip Gram** produced moderately effective results, indicating that while they capture semantic meanings of words, they might not be as effective for this particular regression task as TF-IDF.

Based on the evaluation metrics, **SVR with TF-IDF** is the best performing model and vectorization method for this dataset. It has the lowest Mean Squared Error (MSE) and Root Mean Squared Error (RMSE), indicating its superior ability to predict the scores accurately. This choice is supported by its robust performance metrics, making it the preferred model for this short answer grading task.

![téléchargement (1)](https://github.com/mohammed-stalin/nlp_thirdLab/assets/116387474/414db219-4277-408f-9149-0e3ffc2f00e7)
![téléchargement](https://github.com/mohammed-stalin/nlp_thirdLab/assets/116387474/e543ebad-369a-43c3-a9de-2111a7d4c11e)
![téléchargement (2)](https://github.com/mohammed-stalin/nlp_thirdLab/assets/116387474/988c70fa-667b-42e9-bbe4-6ee2f3f770df)


# Part 2: Language Modeling / Classification

## Dataset
The dataset used for this part can be found here: [Twitter Entity Sentiment Analysis](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

## Preprocessing
Similar to Part 1, an NLP preprocessing pipeline is established for tokenization, stemming, lemmatization, stop words removal, and discretization.

## Encoding
Data vectors are encoded using Word2Vec (CBOW, Skip Gram), Bag of Words, and TF-IDF.

## Models
The following models are trained using the Word2Vec embeddings:

- Support Vector Machine (SVM)
- Naive Bayes
- Logistic Regression
- AdaBoost

## Evaluation
Models are evaluated using standard metrics such as Accuracy, Loss, and F1 Score, along with other metrics like BLEU Score. The best model is selected based on these metrics, and the choice is justified.

## Results Interpretation
The obtained results are interpreted to understand the effectiveness of the selected model and the overall performance of the language models for classification.

# Conclusion
In this project, we explored NLP techniques and models for language modeling using Sklearn. We performed preprocessing, encoding, training, and evaluation for both regression and classification tasks. Through this work, we gained valuable insights into the effectiveness of different models and techniques in NLP applications.
