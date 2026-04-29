# NLP Spam Detection with Explainability

## Project Overview

This project is an end-to-end NLP system that classifies text messages as spam or not spam.

It includes:
- Machine Learning model using TF-IDF + Logistic Regression
- REST API using Flask
- Explainability using important word features
- Docker containerization

---

## Project Structure

- model.py - Train and save model
- explain.py - Explain predictions
- app.py - Flask API
- requirements.txt - Dependencies
- Dockerfile - Container setup
- models/ - Saved model files

---

## Dataset

SMS Spam Classification Dataset

Columns:
- text
- spam

Where:
- 1 = spam
- 0 = not spam

---

## How to Run Locally

```bash
python -m pip install -r requirements.txt
python model.py
python app.py
```

---

## API Usage

### Example Input

```json
{
  "text": "Win free cash now!"
}

```
### Example Output

```json
{
  "prediction": "spam",
  "spam_probability": 0.82,
  "important_words": {
    "free": 2.08,
    "cash": 0.40
  }
}

```

---

## Docker Commands

```bash
docker build -t nlp-app .
docker run -p 5000:5000 nlp-app

```

## Explainability Method

This project uses Logistic Regression feature coefficients to identify the most important words influencing each prediction.


## Author

SYAM B



