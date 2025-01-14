# Research Paper Publishability Prediction System using RAG

## Overview
The **Research Paper Publishability Prediction System** aims to evaluate the publishability of academic papers using Natural Language Processing (NLP) and machine learning techniques. The system predicts whether a paper is publishable and classifies it to the most appropriate conference based on its content. It leverages linguistic, stylistic, and content-based features for classification, combined with a hybrid retrieval approach using **SentenceTransformer** embeddings and **BGE (Bidirectional Generative Embeddings)**.

## Features
- **Publishability Prediction**: Classifies papers as publishable or non-publishable based on specific metrics.
- **Conference Classification**: Recommends the most suitable conferences for a paper using a hybrid retrieval strategy.
- **Explainability**: Provides a rationale for the classification using a generative AI model (Gemini) to ensure transparency.
- **Feature Extraction**: Includes linguistic, stylistic, and content-based features like readability scores, sentence length variations, and lexical density.

## Workflow

### 1. **Feature Extraction**
   The system extracts the following key features from research papers:
   - **Linguistic Features**: Flesch-Kincaid Grade Level, Gunning Fog Index, Coleman-Liau Index, Passive Voice Percentage.
   - **Stylistic Features**: Average Sentence Length, Sentence Length Variation.
   - **Content-Based Features**: Lexical Density, Perplexity.

   The text is extracted from PDF documents using the **PyPDF2** library and preprocessed (tokenization, stop word removal, lemmatization).

### 2. **Dataset Preparation**
   The dataset consists of labeled research papers:
   - **Publishable Papers**: Papers from conferences like CVPR, EMNLP, KDD, NeurIPS, and TMLR.
   - **Non-Publishable Papers**: Papers deemed non-publishable based on specific criteria.
   
   The dataset is processed, cleaned, and labeled, and the extracted features are stored in a CSV file.

### 3. **Model Training**
   The **Logistic Regression** model is used for classification due to its interpretability and suitability for small datasets. The features are scaled using **Min-Max Scaling**, and hyperparameter tuning is performed for optimization.

### 4. **Model Validation**
   Validation strategies include **K-Fold Cross-Validation** and **Test-Train Split**, both yielding perfect scores (F1 Score, Accuracy: 1.00) due to the small dataset and linear separability of features.

### 5. **Research Paper Classification using RAG**
   The classification system uses a hybrid retrieval approach, combining **SentenceTransformer** embeddings with **BGE** for semantic search. The system performs the following steps:
   - **Document Processing**: Extracts text, chunks it, and stores embeddings in a **ChromaDB** vector store.
   - **Ensemble Retrieval**: Retrieves similar papers using both SentenceTransformer and BGE models.
   - **Conference Classification**: The top conference is predicted based on the similarity of papers.
   - **Rationale Generation**: The **Gemini** model provides an explanation for the conference recommendation.

### 6. **Key Features**
   - **Hybrid Search**: Uses both SentenceTransformer and BGE embeddings for accurate classification.
   - **Explainability**: Provides reasoning behind the classification with Gemini model-generated rationales.
   - **Publishability Filtering**: Only processes publishable papers, improving efficiency.

### 7. **Challenges and Future Improvements**
   - **Scalability**: Optimization of the vector store and retrieval process for handling large datasets.
   - **Improved Rationale Generation**: Enhancing the quality of rationale explanations.
   - **Generalization**: Extending the model to classify papers for journals or grants, in addition to conferences.

## Requirements
- Python 3.x
- **Libraries**:
  - PyPDF2
  - SentenceTransformer
  - ChromaDB
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn
