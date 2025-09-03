# d:/Users/Fiona/Desktop/projects/DLProject/src/processing/sentiment_analyzer.py
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence

class SentimentAnalyzer:
    def __init__(self, use_cache: bool = True, models: list = None):
        """
        Initializes the sentiment analysis models with enhanced options.
        
        Args:
            use_cache: Whether to cache model predictions for performance
            models: List of models to use, default is all ["transformer", "vader", "flair"]
        """
        self.use_cache = use_cache
        self.models = models or ["transformer", "vader", "flair"]
        self.cache = {}  # Simple in-memory cache
        
        print("Initializing sentiment analysis models...")
        
        # Load models based on configuration
        if "transformer" in self.models:
            print("Loading Transformer model...")
            # Using a distilled version for speed, can be swapped with other models
            self.transformer_pipeline = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512
            )
            print("Transformer model loaded.")
        
        if "vader" in self.models:
            print("Loading VADER model...")
            self.vader_analyzer = SentimentIntensityAnalyzer()
            print("VADER model loaded.")
        
        if "flair" in self.models:
            print("Loading Flair model...")
            self.flair_classifier = TextClassifier.load('sentiment')
            print("Flair model loaded.")
        
        print("All sentiment models initialized.")
        
    def _get_cache_key(self, text: str, model: str) -> str:
        """Creates a unique cache key for text and model combination."""
        # Simple hash-based cache key
        return f"{hash(text)}-{model}"
    
    def analyze_transformer(self, text: str):
        """
        Analyzes sentiment using a Hugging Face Transformer model with caching.
        """
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        # Check cache first
        cache_key = self._get_cache_key(text, "transformer")
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Handle long texts by truncating if needed
            if len(text) > 1000:
                text = text[:1000]  # The pipeline will handle further truncation
                
            result = self.transformer_pipeline(text)[0]
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = result
                
            return result
        except Exception as e:
            print(f"Error in transformer analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.0}

    def analyze_vader(self, text: str):
        """
        Analyzes sentiment using VADER with caching.
        """
        if not text or not isinstance(text, str):
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        
        # Check cache first
        cache_key = self._get_cache_key(text, "vader")
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = self.vader_analyzer.polarity_scores(text)
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = result
                
            return result
        except Exception as e:
            print(f"Error in VADER analysis: {e}")
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}

    def analyze_flair(self, text: str):
        """
        Analyzes sentiment using Flair with caching.
        """
        if not text or not isinstance(text, str):
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        # Check cache first
        cache_key = self._get_cache_key(text, "flair")
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Handle long texts by splitting into smaller chunks
            if len(text) > 1000:
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                results = []
                
                for chunk in chunks:
                    sentence = Sentence(chunk)
                    self.flair_classifier.predict(sentence)
                    if sentence.labels:
                        results.append((sentence.labels[0].value, sentence.labels[0].score))
                
                if results:
                    # Aggregate results - take the most common sentiment, average the scores
                    labels = [r[0] for r in results]
                    scores = [r[1] for r in results]
                    
                    from collections import Counter
                    most_common_label = Counter(labels).most_common(1)[0][0]
                    avg_score = sum(scores) / len(scores)
                    
                    result = {'label': most_common_label, 'score': avg_score}
                else:
                    result = {'label': 'NEUTRAL', 'score': 0.0}
            else:
                sentence = Sentence(text)
                self.flair_classifier.predict(sentence)
                
                if sentence.labels:
                    result = {'label': sentence.labels[0].value, 'score': sentence.labels[0].score}
                else:
                    result = {'label': 'NEUTRAL', 'score': 0.0}
            
            # Cache result
            if self.use_cache:
                self.cache[cache_key] = result
                
            return result
        except Exception as e:
            print(f"Error in Flair analysis: {e}")
            return {'label': 'NEUTRAL', 'score': 0.0}
            
    def get_normalized_sentiment(self, text: str):
        """
        Returns a normalized sentiment score between -1 and 1 by combining multiple models.
        
        This creates a weighted ensemble of all the models for more robust prediction.
        """
        results = {}
        weights = {"transformer": 0.4, "vader": 0.3, "flair": 0.3}  # Customizable weights
        
        if "transformer" in self.models:
            transformer_result = self.analyze_transformer(text)
            # Convert POSITIVE/NEGATIVE to numeric scale
            if transformer_result['label'] == 'POSITIVE':
                score = transformer_result['score']
            elif transformer_result['label'] == 'NEGATIVE':
                score = -transformer_result['score']
            else:
                score = 0
            results["transformer"] = score
        
        if "vader" in self.models:
            vader_result = self.analyze_vader(text)
            # VADER compound score is already between -1 and 1
            results["vader"] = vader_result['compound']
        
        if "flair" in self.models:
            flair_result = self.analyze_flair(text)
            # Convert POSITIVE/NEGATIVE to numeric scale
            if flair_result['label'] == 'POSITIVE':
                score = flair_result['score']
            elif flair_result['label'] == 'NEGATIVE':
                score = -flair_result['score']
            else:
                score = 0
            results["flair"] = score
        
        # Calculate weighted average
        if results:
            weighted_sum = 0
            total_weight = 0
            
            for model, score in results.items():
                weight = weights.get(model, 1)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
        
        return 0.0  # Neutral if no results


    def analyze_sentiment_on_dataframe(self, df: pd.DataFrame, text_column: str = 'text', 
                                    content_column: str = None, batch_size: int = 10):
        """
        Adds sentiment scores to a DataFrame with advanced options.
        
        Args:
            df: Input DataFrame containing text data
            text_column: Column name containing text to analyze
            content_column: Optional column with full content (e.g., article body)
            batch_size: Number of items to process in parallel
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Add normalized sentiment score (ensemble of all models)
        print(f"Adding normalized sentiment scores...")
        result_df['sentiment_score'] = result_df[text_column].apply(
            lambda x: self.get_normalized_sentiment(x)
        )
        
        # Add sentiment category based on score
        def categorize_sentiment(score):
            if score > 0.2:
                return 'POSITIVE'
            elif score < -0.2:
                return 'NEGATIVE'
            else:
                return 'NEUTRAL'
        
        result_df['sentiment_category'] = result_df['sentiment_score'].apply(categorize_sentiment)
        
        # Add individual model scores if requested
        if "transformer" in self.models:
            print("Adding transformer sentiment scores...")
            result_df['transformer_sentiment'] = result_df[text_column].apply(
                lambda x: self.analyze_transformer(x).get('label')
            )
            result_df['transformer_score'] = result_df[text_column].apply(
                lambda x: self.analyze_transformer(x).get('score')
            )

        if "vader" in self.models:
            print("Adding VADER sentiment scores...")
            result_df['vader_sentiment'] = result_df[text_column].apply(
                lambda x: self.analyze_vader(x).get('compound')
            )

        if "flair" in self.models:
            print("Adding Flair sentiment scores...")
            result_df['flair_sentiment'] = result_df[text_column].apply(
                lambda x: self.analyze_flair(x).get('label')
            )
            result_df['flair_score'] = result_df[text_column].apply(
                lambda x: self.analyze_flair(x).get('score')
            )
        
        # If content column is provided, analyze full content
        if content_column and content_column in df.columns:
            print(f"Analyzing full content from '{content_column}'...")
            # Only analyze non-null content
            mask = df[content_column].notna()
            if mask.any():
                content_df = df[mask].copy()
                content_df['content_sentiment'] = content_df[content_column].apply(
                    lambda x: self.get_normalized_sentiment(x)
                )
                
                # Merge content sentiment back to main dataframe
                result_df = result_df.merge(
                    content_df[['content_sentiment']], 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
        
        # Add metadata timestamp
        from datetime import datetime
        result_df['analysis_timestamp'] = datetime.now().isoformat()
        
        return result_df

if __name__ == '__main__':
    # Example Usage
    analyzer = SentimentAnalyzer()
    
    # From a CSV (e.g., generated by a scraper)
    try:
        tweets_df = pd.read_csv("twitter_feed.csv")
        if 'text' in tweets_df.columns:
            tweets_df = analyzer.analyze_sentiment_on_dataframe(tweets_df, text_column='text')
            print("Sentiment analysis on tweets:")
            print(tweets_df.head())
            tweets_df.to_csv("tweets_with_sentiment.csv", index=False)
            print("Saved tweets with sentiment to tweets_with_sentiment.csv")

    except FileNotFoundError:
        print("twitter_feed.csv not found. Running with sample data.")
        # Sample data
        data = {'text': [
            "Stocks are going to the moon!", 
            "I am feeling bearish about the market.",
            "The new earnings report is just okay."
        ]}
        sample_df = pd.DataFrame(data)
        sample_df = analyzer.analyze_sentiment_on_dataframe(sample_df, text_column='text')
        print("\nSentiment analysis on sample data:")
        print(sample_df.head())

