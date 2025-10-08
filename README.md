# StackOne: Technology Domain Classification

A machine learning model that classifies Stack Overflow questions into technology domains based on tags and metadata.

Dataset link: https://www.kaggle.com/datasets/stackoverflow/stacklite
## Model Performance
- **Accuracy**: 99%
- **Classes**: 6 technology domains
- **Dataset**: 48K Stack Overflow questions

## Technology Domains
- **Backend/Systems**: Java, C#, C++, Go, Rust
- **Web Development**: HTML, CSS, JavaScript, React, Angular
- **Data Science**: Python, ML, Pandas, TensorFlow
- **Database**: SQL, MySQL, PostgreSQL, MongoDB
- **DevOps/Cloud**: Docker, AWS, Kubernetes, Linux
- **Mobile Development**: Android, iOS, Swift, Kotlin

## Quick Start

### Training
```bash
python tech_domain_classifier.py
```

### Prediction
```python
from predict_domain import predict_domain

domain, confidence = predict_domain(['python', 'pandas'], score=10, answer_count=2)
print(f"Domain: {domain}")  # Output: Data Science
```

## Files
- `tech_domain_classifier.py` - Model training
- `predict_domain.py` - Prediction interface
- `tech_domain_model.pkl` - Trained model
- `data/` - Stack Overflow dataset

## Requirements
```
pandas
scikit-learn
numpy
```

## Dataset Structure
- `questions.csv`: Question metadata (ID, score, answers, dates)
- `question_tags.csv`: Tag associations (ID, tag pairs)

## Features
- Tag presence indicators (top 50 tags)
- Question score and answer count
- Score-to-answer ratio
- Creation year
- Tag count per question
