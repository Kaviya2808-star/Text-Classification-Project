# Text Classification Using Naive Bayes

## Project Overview
This project demonstrates a text classification system using Machine Learning. The model is trained to classify text documents into different categories based on their content.

## Dataset
The dataset contains two columns:
- Text: The document or sentence to classify.
- Category: The corresponding class label.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

## Project Workflow

### Step 1: Import Libraries
Required libraries are imported for data processing, feature extraction, model training, and evaluation.

### Step 2: Load Dataset
The dataset is loaded using Pandas.

### Step 3: Data Preprocessing
Missing category values are removed using:
```python
df = df.dropna(subset=["Category"])
```

### Step 4: Feature Extraction
Text data is converted into numerical form using TF-IDF Vectorization.

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
```

### Step 5: Train-Test Split
The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 6: Model Training
A Multinomial Naive Bayes model is trained using the training data.

```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

### Step 7: Prediction
The trained model predicts categories for the test data.

```python
y_pred = model.predict(X_test)
```

### Step 8: Evaluation
Model performance is evaluated using accuracy score.

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Results
The model successfully classifies text documents into their respective categories and provides prediction accuracy.

## Author
Kaviya M
