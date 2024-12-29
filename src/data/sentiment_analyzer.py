import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
import os
from typing import Dict, Optional

class SentimentAnalyzer:
    """Analyzes market sentiment using multiple data sources."""
    
    def __init__(self):
        self.fear_greed_url = os.getenv('FEAR_GREED_API_URL', 'https://api.alternative.me/fng/')
        self.fear_greed_cache: Dict[str, float] = {}
        self.social_sentiment_cache: Dict[str, Dict[str, float]] = {}
    
    async def get_fear_greed_index(self, date: datetime) -> float:
        """Get Fear & Greed index for a specific date."""
        try:
            # Check cache first
            date_str = date.strftime('%Y-%m-%d')
            if date_str in self.fear_greed_cache:
                return self.fear_greed_cache[date_str]
            
            # Fetch from API
            async with aiohttp.ClientSession() as session:
                params = {
                    'date': date_str,
                    'limit': 1
                }
                async with session.get(self.fear_greed_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and data['data']:
                            value = float(data['data'][0]['value'])
                            self.fear_greed_cache[date_str] = value
                            return value
            
            # Return neutral value if API fails
            return 50.0
            
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed index: {e}")
            return 50.0
    
    async def get_social_sentiment(self, symbol: str, date: datetime) -> Dict[str, float]:
        """Get social media sentiment for a specific symbol and date."""
        try:
            # This would normally fetch from a social media API
            # For now, return mock data based on symbol and date
            date_str = date.strftime('%Y-%m-%d')
            cache_key = f"{symbol}_{date_str}"
            
            if cache_key in self.social_sentiment_cache:
                return self.social_sentiment_cache[cache_key]
            
            # Mock sentiment scores (would be replaced with real API calls)
            sentiment = {
                'twitter_score': 0.5 + (hash(cache_key) % 100) / 200,  # Random between 0 and 1
                'reddit_score': 0.5 + (hash(cache_key + '_reddit') % 100) / 200,
                'overall_score': 0.5  # Neutral default
            }
            
            self.social_sentiment_cache[cache_key] = sentiment
            return sentiment
            
        except Exception as e:
            logger.error(f"Error getting social sentiment for {symbol}: {e}")
            return {'twitter_score': 0.5, 'reddit_score': 0.5, 'overall_score': 0.5}
    
    async def get_combined_sentiment(self, symbol: str, date: datetime) -> float:
        """Get combined sentiment score from all sources."""
        try:
            fear_greed = await self.get_fear_greed_index(date)
            social = await self.get_social_sentiment(symbol, date)
            
            # Normalize fear & greed to 0-1 range
            fear_greed_normalized = fear_greed / 100.0
            
            # Combine different signals (can be adjusted)
            weights = {
                'fear_greed': 0.4,
                'twitter': 0.3,
                'reddit': 0.3
            }
            
            combined_score = (
                weights['fear_greed'] * fear_greed_normalized +
                weights['twitter'] * social['twitter_score'] +
                weights['reddit'] * social['reddit_score']
            )
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error calculating combined sentiment: {e}")
            return 0.5  # Neutral sentiment on error 