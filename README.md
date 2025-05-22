# sentiment_analysis_of_bd_banking_apps_user_reviews
## Project Title: Sentiment Detection of User Reviews for Bangladeshi Banking Apps

## Dataset 
The dataset was collected through
 web scraping from the Google Play Store. Initially, the reviews
 for 17 Bangladeshi apps were collected and merged into a
 csv file. The 78938 user reviews were collected and from
 them 1618 reviews were labeled manually and the rest of
 them remain unlabeled due to shortage of time. This dataset
 contains reviews with Bengali and English and mixed of both
 languages. The reviews were labeled as positive or negative
 only. 
## Data Preprocessing
1. Data Cleaning
2. Tokenization 
3. Stopword Removal
4. Stemming 
5. Language Detection

Accuracy, Precision, Recall and F1-Score were used for evaluation of models. 
## Model Training & Evaluation
The following machine learning algorithms were applied:
1. Naive Bayes
2. Support Vector Machine (SVM)
3. Random Forest
4. Decision Tree
5. Logistic Regression
   
## Hyperparameter Tuning
Hyperparameter optimization was performed with GridSearchCV for Naive Bayes and Random Forest (Achieved the highest accuracy compared to other algorithms).

## Results
Naive Bayes achieved the highest accuracy among all the models tested.
