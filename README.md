# Natural Language Processing with Disaster Tweets

A comprehensive machine learning project for binary classification of disaster-related tweets using advanced NLP techniques and deep learning models.

## 📋 Project Overview

This project addresses the Kaggle competition "Natural Language Processing with Disaster Tweets" which involves building a machine learning model to predict whether a given tweet is about a real disaster or not. This is a binary text classification problem with significant real-world applications in emergency response and social media monitoring.

### Business Context

Twitter has become an important communication channel during emergencies, with people sharing real-time information about disasters. Automated systems that can quickly identify disaster-related tweets can help emergency services respond faster and more effectively. The ability to distinguish between actual disaster reports and metaphorical or non-urgent uses of disaster-related language is crucial for effective emergency response.

## 🎯 Problem Statement

- **Task**: Binary classification (Disaster: 1, Not Disaster: 0)
- **Training Set**: ~7,613 hand-labeled tweets
- **Test Set**: ~3,263 tweets (unlabeled)
- **Challenge**: Distinguish between literal and metaphorical uses of disaster-related language

## 📊 Dataset Description

The dataset includes the following features:
- `id`: Unique identifier for each tweet
- `text`: The actual tweet text
- `location`: Location information (may be blank)
- `keyword`: Disaster-related keyword (may be blank)
- `target`: Binary label (1 = disaster, 0 = not disaster) - only in training data

## 🛠️ Technical Approach

### 1. Data Preprocessing
- **Text Cleaning**: URL removal, hashtag processing, mention handling
- **Tokenization**: NLTK-based tokenization with custom preprocessing
- **Lemmatization**: WordNet lemmatizer for word normalization
- **Feature Engineering**: 31+ engineered features including:
  - Text statistics (length, word count, etc.)
  - Punctuation and style features
  - Social media specific features (mentions, hashtags, URLs)
  - Disaster-related keyword presence
  - Sentiment indicators

### 2. Text Vectorization
- **TF-IDF Vectorization**: Term frequency-inverse document frequency
- **Keras Tokenization**: For neural network input
- **Word Embeddings**: Pre-trained embeddings setup for semantic understanding

### 3. Model Architecture

Multiple neural network architectures were implemented and compared:

#### Simple LSTM
- Single LSTM layer with dropout
- Baseline sequential model

#### Bidirectional LSTM
- Processes text in both directions
- Better context understanding
- Captures dependencies from past and future tokens

#### GRU (Gated Recurrent Unit)
- Simplified version of LSTM with fewer parameters
- Faster training while maintaining performance
- **Best performing model** with F1-score of 0.77

#### CNN-LSTM Hybrid
- Combines Convolutional Neural Networks with LSTM
- CNN extracts local patterns, LSTM captures sequential dependencies

#### Multi-Input Model
- Combines text embeddings with engineered features
- Separate branches for text and numerical features
- Fusion layer for final classification

### 4. Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Callbacks**: Early stopping and learning rate reduction
- **Validation Split**: 20% of training data

## 🏆 Results

### Model Performance Comparison

| Model | Validation Accuracy | F1-Score | Training Time |
|-------|-------------------|----------|---------------|
| Simple LSTM | 57.06% | 0.00 | 864s |
| **GRU** | **79.91%** | **0.77** | **542s** |
| Bidirectional LSTM | 79.65% | 0.77 | 575s |
| CNN-LSTM | 78.00% | 0.77 | 316s |
| Multi-Input | 80.17% | 0.77 | 581s |

### Best Model: GRU
- **Accuracy**: 79.91%
- **F1-Score**: 0.7685
- **AUC-ROC**: 0.8706
- **Precision**: 0.76 (Disaster class)
- **Recall**: 0.78 (Disaster class)

## 🚀 Getting Started

### Prerequisites
- Python 3.7+ (recommended: Python 3.8+)
- GPU support recommended for training (CUDA-compatible)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Natural-Language-Processing-with-Disaster-Tweets
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv disaster_tweets_env
   source disaster_tweets_env/bin/activate  # On Windows: disaster_tweets_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if not automatically downloaded)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

### Dataset Setup

1. **Download the Kaggle dataset**
   - Go to [Kaggle Competition Page](https://www.kaggle.com/c/nlp-getting-started)
   - Download `train.csv`, `test.csv`, and `sample_submission.csv`
   - Update the file paths in the notebook if needed

2. **Alternative**: Update data paths in the notebook
   ```python
   TRAIN_PATH = 'path/to/your/train.csv'
   TEST_PATH = 'path/to/your/test.csv'
   SAMPLE_SUBMISSION_PATH = 'path/to/your/sample_submission.csv'
   ```

## 💻 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   ```
   Natural-Language-Processing-with-Disaster-Tweets-Notebook.ipynb
   ```

3. **Execute cells sequentially** or run all cells

### Key Sections

1. **Data Setup and EDA** - Explore the dataset and understand patterns
2. **Text Preprocessing** - Clean and prepare text data
3. **Feature Engineering** - Extract numerical features from text
4. **Model Training** - Train multiple neural network architectures
5. **Evaluation** - Compare models and analyze results
6. **Submission** - Generate Kaggle submission file

### Quick Run

For a quick test run, you can:
1. Load the preprocessed features
2. Train a single model (e.g., GRU)
3. Generate predictions

## 📈 Model Training

### Training Process
1. **Data Preprocessing**: Text cleaning and feature extraction
2. **Vectorization**: Convert text to numerical representations
3. **Model Creation**: Initialize neural network architectures
4. **Training**: Fit models with validation monitoring
5. **Evaluation**: Compare performance metrics
6. **Selection**: Choose best performing model

### Hyperparameter Tuning
The notebook includes optimized hyperparameters:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 15 (with early stopping)
- Dropout rates: 0.3-0.5
- Embedding dimension: 100

## 🔍 Key Features

### Text Preprocessing Pipeline
- URL and mention removal
- Hashtag processing
- Contraction expansion
- Lemmatization
- Stopword filtering (with disaster word preservation)

### Feature Engineering
- **Statistical Features**: Text length, word count, character count
- **Style Features**: Punctuation, capitalization, exclamation marks
- **Content Features**: Disaster keywords, sentiment words
- **Social Media Features**: URLs, mentions, hashtags

### Model Innovation
- Bidirectional processing for better context
- Multi-input architecture combining text and features
- Custom F1-score optimization
- Ensemble-ready architecture

## 🎯 Business Impact

### Applications
- **Emergency Response**: Rapid identification of real disasters
- **Social Media Monitoring**: Filter noise from actual emergencies
- **Resource Allocation**: Direct emergency services efficiently
- **Public Safety**: Early warning system development

### Performance Metrics
- **Precision**: Minimizes false alarms (important for resource allocation)
- **Recall**: Catches actual disasters (critical for public safety)
- **F1-Score**: Balanced measure for both precision and recall

## 🛡️ Error Analysis

The notebook includes comprehensive error analysis:
- **False Positives**: Non-disasters predicted as disasters
- **False Negatives**: Actual disasters missed by the model
- **Confidence Analysis**: Model uncertainty quantification
- **Threshold Optimization**: Adjusting classification threshold for better F1-score

## 📁 File Structure

```
Natural-Language-Processing-with-Disaster-Tweets/
├── Natural-Language-Processing-with-Disaster-Tweets-Notebook.ipynb  # Main notebook
├── requirements.txt                                                 # Python dependencies
├── README.md                                                       # This file
├── disaster_tweets_submission.csv                                  # Generated submission file
└── data/                                                          # Dataset directory (create manually)
    ├── train.csv                                                  # Training data
    ├── test.csv                                                   # Test data
    └── sample_submission.csv                                      # Submission format
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Kaggle** for providing the dataset and competition platform
- **TensorFlow/Keras** for deep learning framework
- **NLTK** for natural language processing tools
- **Scikit-learn** for machine learning utilities
- **Open source community** for various libraries and tools

## 📞 Contact

For questions, suggestions, or collaborations:
- Create an issue in this repository
- Contact through Kaggle platform

---

**Note**: This project demonstrates advanced NLP techniques for disaster tweet classification. The methodology can be adapted for other text classification tasks in emergency response, social media analysis, and crisis management domains.