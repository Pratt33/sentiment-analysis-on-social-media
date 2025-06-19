# Sentiment Analysis on Social Media

## Project Overview
This project performs sentiment analysis on tweets using machine learning models. It includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and a Streamlit web app for inference.

## Directory Structure
```
./
├── data/                 # Place datasets here (not tracked by git)
├── notebooks/            # EDA and experimentation notebooks
├── src/                  # Scripts: preprocess, model, evaluate
├── app/                  # Streamlit app
├── output/               # Model files, vectorizer (smallest only)
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── LICENSE               # License info
├── .gitignore            # Ignore rules for sensitive/large files
```

## Usage
1. **Download the Sentiment140 dataset** from [Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140) and place the CSV in `data/`.
2. **Train models:**
   ```bash
   python src/train.py data/training.1600000.processed.noemoticon.csv output
   ```
3. **Evaluate and select best model:**
   ```bash
   python src/evaluate.py output data/training.1600000.processed.noemoticon.csv
   ```
4. **Run the Streamlit app:**
   ```bash
   streamlit run app/app.py
   ```

## Security & Best Practices
- **Never commit API keys (e.g., `kaggle.json`) or large data files.**
- The `.gitignore` is set to exclude sensitive and large files by default.
- Only small, necessary model files are kept in `output/` for deployment.

## Example
- Enter a tweet in the app and get a sentiment prediction (Positive/Negative).

---
For more details, see the code and comments in each script. 