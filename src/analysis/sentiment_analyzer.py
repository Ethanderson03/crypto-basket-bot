from datetime import datetime
from loguru import logger
import aiohttp

class SentimentAnalyzer:
    def __init__(self):
        self.fear_greed_cache = {}
        self.fear_greed_url = "https://api.alternative.me/fng/"
        
    async def analyze_sentiment(self) -> float:
        """Analyze market sentiment and return a multiplier (0.5 to 2.0)."""
        try:
            # Get Fear & Greed Index
            fear_greed = await self.get_fear_greed_index()
            
            # Enhanced contrarian logic
            if fear_greed <= 25:
                # Extreme fear - stronger contrarian signal
                base_mult = 1.5 + (25 - fear_greed) / 25  # 1.5 to 2.0
            elif fear_greed >= 75:
                # Extreme greed - stronger contrarian signal
                base_mult = 1.0 - (fear_greed - 75) / 50  # 0.5 to 1.0
            else:
                # Normal range - linear scaling
                base_mult = 1.0 + (50 - fear_greed) / 100  # 0.5 to 1.5
            
            # Ensure within bounds
            return max(min(base_mult, 2.0), 0.5)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 1.0  # Neutral sentiment as fallback
            
    async def get_fear_greed_index(self) -> float:
        """Get the current Fear & Greed Index value (0-100)."""
        try:
            # Check cache first
            now = datetime.now()
            cache_key = now.strftime("%Y-%m-%d")
            if cache_key in self.fear_greed_cache:
                return self.fear_greed_cache[cache_key]
            
            # In backtest mode, return a simulated value
            # This is a simplified simulation - in a real implementation,
            # you would use historical Fear & Greed Index data
            simulated_value = 50.0  # Neutral value
            self.fear_greed_cache[cache_key] = simulated_value
            return simulated_value
            
        except Exception as e:
            logger.error(f"Error getting Fear & Greed Index: {str(e)}")
            return 50.0  # Neutral value as fallback 