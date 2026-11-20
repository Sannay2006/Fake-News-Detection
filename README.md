# Fake News Detection Project

## Local web app

The repository includes a responsive article-analysis app that runs entirely on your machine. It uses the TF-IDF + Logistic Regression approach from the notebook on `Fake.csv` (class `0`, fake) and `True.csv` (class `1`, real) inside `Datasets/datasets.zip`.

Requires Python 3.10 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. On the first analysis, the app trains a balanced sample of up to 6,000 articles per class and saves the pipeline to `model/fake_news_pipeline.joblib`. The model cache is ignored by Git; remove it to train again.

`GET /api/health` reports whether the model is loaded. Send an article of 60–20,000 characters to `POST /api/analyze` as JSON:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/analyze `
  -Method Post -ContentType 'application/json' `
  -Body '{"article":"Paste at least 60 characters of article text here..."}'
```

The response contains a `label` (`fake` or `real`), a `confidence` score from 0 to 1, up to five contributing text `signals`, and a caveat. Confidence is the model's relative class score, not the probability that a claim is true. This tool detects patterns in historical training data; it does not fact-check articles.

Run the app tests with `python -m pytest`.

## Deploy on Render

The repository includes a Render Blueprint in `render.yaml`. To publish the app, connect the public GitHub repository to Render and create a Blueprint instance from that file. Render builds the Python app, serves it with Waitress, and checks `/api/health`. The first article analyzed trains the model from the bundled dataset; free instances may sleep when idle and retrain after a restart.

<img src="https://socialify.git.ci/kapilsinghnegi/Fake-News-Detection/image?description=1&font=Source%20Code%20Pro&forks=1&issues=1&language=1&name=1&owner=1&pattern=Charlie%20Brown&pulls=1&stargazers=1&theme=Dark" alt="Fake-News-Detection" width="1280" height="320" />


The project aims to develop a machine-learning model capable of identifying and classifying any news article as fake or not. The distribution of fake news can potentially have highly adverse effects on people and culture. This project involves building and training a model to classify news as fake news or not using a diverse dataset of news articles. We have used four techniques to determine the results of the model.

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Gradient Boost Classifier**
4. **Random Forest Classifier**

## Project Overview

Fake news has become a significant issue in today's digital age, where information spreads rapidly through various online platforms. This project leverages machine learning algorithms to automatically determine the authenticity of news articles, providing a valuable tool to combat misinformation.

## Dataset

We have used a labelled dataset containing news articles along with their corresponding labels (true or false). The dataset is divided into two classes:
- True: Genuine news articles
- False: Fake or fabricated news articles

## System Requirements

- Python 3.10 or later
- Several hundred megabytes of disk space for the dataset and Python dependencies

## Dependencies

The web app's dependencies are listed in `requirements.txt`; install them with `python -m pip install -r requirements.txt`.

## Usage

Run `python app.py` and open <http://127.0.0.1:5000> to analyze an article. The original notebook remains available for exploring its classifiers.

## Results

We evaluated each classifier's performance using metrics such as accuracy, precision, recall, and F1 score. The results are documented in the project files.

## Model Deployment

Once you are satisfied with the performance of a particular classifier, you can deploy it in a real-world application or integrate it into a larger system for automatic fake news detection.
---

## Project Screenshots

#### Not a Fake News
![Not a Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/3d079c46-118a-4c53-a515-43b9146001c5)

#### Fake News
![Fake News](https://github.com/kapilsinghnegi/Fake-News-Detection/assets/118688453/2f5262f7-801d-4293-824c-13c29fb97fed)
