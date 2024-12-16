import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def preprocess_text(text):
    """Clean and preprocess the input text."""
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[\^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'@[A-Za-z0–9]+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove the '#' symbol
    text = re.sub(r'RT[\s]+', '', text)  # Remove RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
    text = re.sub(r':', '', text)  # Remove ':' symbol
    text = re.sub(r'[\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    return ' '.join(words)

def calculate_sentiment_score(text):
    """Calculate the sentiment score for a given text."""
    max_tokens = tokenizer.model_max_length
    tokens = tokenizer.tokenize(text)

    # Split the text into chunks to respect the token limit
    chunks = [" ".join(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

    scores = {"positive": 0, "negative": 0, "neutral": 0}
    total_chunks = len(chunks)

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_tokens)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Add the chunk scores
        scores["positive"] += probabilities[0][0].item()
        scores["negative"] += probabilities[0][1].item()
        scores["neutral"] += probabilities[0][2].item()

    if total_chunks == 0:
        return {"positive": 0, "negative": 0, "neutral": 0}

    # Normalize the scores by the number of chunks
    for key in scores:
        scores[key] /= total_chunks

    return scores

def analyze_sentiments(texts, mode='sep'):
    """Analyze the sentiments of a list of texts.

    Args:
        texts (list): List of texts to analyze.
        mode (str): Analysis mode, 'sep' for separate scores, 'agg' for aggregated value.

    Returns:
        list: if mode='sep', a list of tuples with (negative, neutral, positive) scores.
        list: if mode='agg', a list of aggregated sentiment scores between -1 (negative) and 1 (positive).
    """
    results = []

    for text in texts:
        scores = calculate_sentiment_score(text)
        neg, neut, pos = scores['negative'], scores['neutral'], scores['positive']

        if mode == 'sep':
            results.append((neg, neut, pos))
        elif mode == 'agg':
            aggregated_score = (-neg + pos) / (neg + neut + pos)
            results.append(aggregated_score)
        else:
            raise ValueError("Invalid mode. Choose 'sep' or 'agg'.")

    return results

# Load the tokenizer and model
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "The market is going up. Investors are optimistic.",
        "The FED announcement caused a lot of uncertainty.",
        "The situation is neutral, no big changes expected."
    ]

    # Sentiment analysis in 'sep' mode
    print("Separate sentiment scores:")
    print(analyze_sentiments(sample_texts, mode='sep'))

    # Sentiment analysis in 'agg' mode
    print("Aggregated sentiment scores:")
    print(analyze_sentiments(sample_texts, mode='agg'))
