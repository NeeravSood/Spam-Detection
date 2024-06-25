import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def spam_detector(train_df, valid_df, test_df):
    # Step 1: Perform Tf-idf vectorization on text data
    tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    
    # Fit Tf-idf vectorizer on training data and transform all datasets
    X_train = tfidf_vectorizer.fit_transform(train_df['text'])
    X_valid = tfidf_vectorizer.transform(valid_df['text'])
    X_test = tfidf_vectorizer.transform(test_df['text'])
    
    # Prepare labels
    y_train = train_df['label']
    y_valid = valid_df['label']
    
    # Step 2: Initialize classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=0),
        'MultinomialNB': MultinomialNB(),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0),
        'LinearSVC': LinearSVC(random_state=0)
    }
    
    results = {}
    best_classifier = None
    min_spam_misclassified = float('inf')
    
    # Step 3: Train and validate each classifier
    for clf_name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred_valid = clf.predict(X_valid)
        
        # Calculate confusion matrix for validation data
        cm = confusion_matrix(y_valid, y_pred_valid)
        
        # Save confusion matrix and model for the classifier
        results[clf_name] = {
            'confusion_matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
            'model': clf
        }
        
        # Ensure confusion matrix has correct shape
        assert cm.shape == (2, 2), f"Confusion matrix shape is not (2, 2) for {clf_name}: {cm.shape}"
        
        # Determine the number of spam messages misclassified as ham
        spam_misclassified_as_ham = cm[0][1]  # false positives (spam misclassified as ham)
        
        # Check if this classifier has the lowest misclassification rate for spam
        if spam_misclassified_as_ham < min_spam_misclassified:
            min_spam_misclassified = spam_misclassified_as_ham
            best_classifier = clf_name
    
    # Step 4: Prepare final output
    best_model = results[best_classifier]['model']
    results['BestClassifier'] = best_classifier
    results['TfidfVectorizer'] = tfidf_vectorizer
    results['Prediction'] = best_model.predict(X_test).tolist()  # Convert numpy array to list
    
    # Remove models from the results to prevent serialization issues
    for clf_name in classifiers.keys():
        del results[clf_name]['model']
    
    return results

# Example usage (commented out since we do not have actual data here):
# train_df = pd.DataFrame({'label': [1, 0, 1], 'text': ["hello", "buy now", "hi"]})
# valid_df = pd.DataFrame({'label': [1, 0], 'text': ["greetings", "cheap meds"]})
# test_df = pd.DataFrame({'label': [1, 0], 'text': ["howdy", "free offer"]})
# results = spam_detector(train_df, valid_df, test_df)
# print(results)
