
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

# Data Loading
def load_data(filepath="/content/IMDB Dataset.csv"):
    df = pd.read_csv(filepath, engine='python')
    return df

# Data Preprocessing
def preprocess_text(df):
    # Remove duplicates
    df = df.drop_duplicates().copy()

    # Function to remove URLs
    def remove_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    # Function to remove numbers
    def remove_numbers(text):
        number_pattern = re.compile(r'\\d+')
        return number_pattern.sub(r'', text)

    # Function to remove extra spaces
    def remove_extra_spaces(text):
        space_pattern = re.compile(r'\\s+')
        return space_pattern.sub(r' ', text).strip()

    # Apply the cleaning functions to the 'review' column
    df['review'] = df['review'].apply(remove_urls)
    df['review'] = df['review'].apply(remove_numbers)
    df['review'] = df['review'].apply(remove_extra_spaces)

    # Convert text to lowercase
    df['review'] = df['review'].str.lower()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    def remove_stopwords(text):
        tokens = text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)
    df['review'] = df['review'].apply(remove_stopwords)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    def lemmatize_text(text):
        tokens = text.split()
        lemmas = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmas)
    df['review'] = df['review'].apply(lemmatize_text)

    return df

# Model Training and Saving
def train_and_save_models(df):
    X = df['review']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    joblib.dump(vectorizer, 'count_vectorizer.pkl')
    print("CountVectorizer saved as 'count_vectorizer.pkl'")

    # Multinomial Naive Bayes Model
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectors, y_train)
    joblib.dump(nb_model, 'multinomial_nb_model.pkl')
    print("Multinomial Naive Bayes model saved as 'multinomial_nb_model.pkl'")

    y_pred_nb = nb_model.predict(X_test_vectors)
    print("\nMultinomial Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))

    # Logistic Regression Model
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vectors, y_train)
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    print("Logistic Regression model saved as 'logistic_regression_model.pkl'")

    y_pred_lr = lr_model.predict(X_test_vectors)
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))

    # Random Forest Classifier Model with Hyperparameter Tuning
    param_grid = {
        'n_estimators': [50, 100],  # Reduced for quicker execution in script
        'max_depth': [None, 10],   # Reduced for quicker execution in script
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=2, # Reduced for quicker execution in script
        scoring='accuracy',
        n_jobs=-1,
        verbose=0 # Set to 0 to reduce verbosity in script
    )
    grid_search.fit(X_train_vectors, y_train)
    best_rf_model = grid_search.best_estimator_
    joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
    print("Best Random Forest Classifier model saved as 'best_random_forest_model.pkl'")

    y_pred_best_rf = best_rf_model.predict(X_test_vectors)
    print("\nBest Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_best_rf))

if __name__ == "__main__":
    print("Starting model building process...")
    data_df = load_data()
    cleaned_df = preprocess_text(data_df)
    train_and_save_models(cleaned_df)
    print("Model building process completed.")
