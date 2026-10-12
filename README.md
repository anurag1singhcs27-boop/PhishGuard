# PhishGuard — beginner starter project

**Project:** Hybrid ML-NLP Model for Real-Time Phishing Email and URL Classification  
**Team:** Payal Jha and Anurag Singh  
**Guide:** Dr. Saurabh Kumar

This starter aligns with the Phase I synopsis and Phase II design: email NLP with TF-IDF and Logistic Regression, URL lexical/structural features with Random Forest, a Streamlit interface, and accuracy/precision/recall/F1/confusion matrices.

## Important limitation
The bundled CSV files are small illustrative examples created to make the app runnable. They are not representative public cybersecurity datasets. Do not claim their metrics represent real-world performance. Before final evaluation, replace them with properly sourced labelled datasets, record the source/licence and preprocessing, and rerun the experiments.

## Run locally on Windows
1. Install Python 3.11 from https://www.python.org/downloads/ and tick **Add Python to PATH** during installation.
2. Install VS Code from https://code.visualstudio.com/ (optional but recommended).
3. Extract this ZIP into a simple folder, e.g. Documents/PhishGuard_Starter.
4. Open that folder in VS Code. Choose **Terminal → New Terminal**.
5. Run these commands one at a time:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

A browser should open at `http://localhost:8501`. If the `streamlit` command is not found, try `python -m streamlit run app.py` after installing requirements.

## Publish with GitHub + Streamlit Community Cloud
1. Create a new public GitHub repository, e.g. `phishguard-phishing-detection`.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder with both CSVs. Preserve the folder structure.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select the repository and `main` branch, and set main file path to `app.py`.
5. Deploy. Once the app is running, copy the generated `https://....streamlit.app` URL into the Google Form's implementation/demo link field.
6. Test the deployed app from a private/incognito browser and your phone before submitting.

## Suggested repository description
Student capstone prototype combining TF-IDF-based email text classification and lexical URL feature extraction with machine-learning models, exposed through a Streamlit interface. Includes illustrative demo data; robust evaluation requires properly sourced labelled datasets.

## Google Form submission
- **Implementation / working project link:** deployed Streamlit app URL, only after it opens and works.
- **Source code / GitHub link:** public repository URL.
- **Dataset link:** the actual source URL of the public dataset you adopt; do not call the bundled illustrative CSV an external public dataset.
- **Evidence:** screenshots of the running app, email analysis, URL analysis, and metrics after actually testing them.

## Viva explanation
“This prototype has two separate classification pipelines under one Streamlit interface. The email pipeline uses TF-IDF to represent text numerically and Logistic Regression for classification. The URL pipeline extracts lexical and structural URL features and uses a Random Forest classifier. The interface displays first-level predictions and evaluation metrics. The included dataset is illustrative; robust evaluation requires a properly sourced, larger labelled dataset.”

## Next work before final evaluation
1. Replace the illustrative CSVs with properly sourced public datasets.
2. Record dataset sources, licences, row counts, labels and class distribution.
3. Inspect missing values, duplicates, invalid rows and label consistency.
4. Evaluate on a held-out test set; consider stratified cross-validation.
5. Compare additional models such as Linear SVM and, if feasible, XGBoost.
6. Save actual metric tables and confusion matrices.
7. Update the report so claims match the code and evidence.
