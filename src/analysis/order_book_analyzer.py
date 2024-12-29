from typing import Dict, List, Optional
from loguru import logger

class OrderBookAnalyzer:
    """Handles detailed order book analysis and execution price calculations."""
    
    def __init__(
        self,
        max_spread_bps: int = 10,        # Maximum allowed spread in basis points
        max_slippage_bps: int = 20,      # Maximum allowed slippage in basis points
        min_depth_usd: float = 100000,   # Minimum depth required at best bid/ask
        impact_threshold_bps: int = 30    # Max allowed market impact in basis points
    ):
        self.max_spread_bps = max_spread_bps
        self.max_slippage_bps = max_slippage_bps
        self.min_depth_usd = min_depth_usd
        self.impact_threshold_bps = impact_threshold_bps
    
    def calculate_spread(self, order_book: Dict) -> Dict:
        """Calculate current spread metrics."""
        try:
            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                return {
                    'absolute_spread': float('inf'),
                    'relative_spread': float('inf'),
                    'bid_depth': 0.0,
                    'ask_depth': 0.0,
                    'is_tradeable': False
                }
            
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            bid_size = float(order_book['bids'][0][1])
            ask_size = float(order_book['asks'][0][1])
            
            absolute_spread = best_ask - best_bid
            relative_spread = absolute_spread / best_bid * 10000  # Convert to basis points
            
            bid_depth = bid_size * best_bid  # Convert to USD
            ask_depth = ask_size * best_ask  # Convert to USD
            
            is_tradeable = (
                relative_spread <= self.max_spread_bps and
                min(bid_depth, ask_depth) >= self.min_depth_usd
            )
            
            return {
                'absolute_spread': absolute_spread,
                'relative_spread': relative_spread,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'is_tradeable': is_tradeable
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread: {e}")
            return {
                'absolute_spread': float('inf'),
                'relative_spread': float('inf'),
                'bid_depth': 0.0,
                'ask_depth': 0.0,
                'is_tradeable': False
            }
    
    def calculate_slippage(
        self,
        order_book: Dict,
        side: str,
        size: float
    ) -> Dict:
        """Calculate expected slippage for an order."""
        try:
            if not order_book:
                return {
                    'expected_average_price': 0.0,
                    'price_impact': float('inf'),
                    'levels_consumed': 0,
                    'unfilled_size': size,
                    'is_executable': False
                }
            
            levels = order_book['asks'] if side == 'buy' else order_book['bids']
            best_price = float(levels[0][0])
            
            remaining_size = size
            weighted_sum = 0.0
            filled_size = 0.0
            levels_consumed = 0
            
            for level in levels:
                price = float(level[0])
                available = float(level[1])
                
                if remaining_size <= 0:
                    break
                
                fill_size = min(remaining_size, available)
                weighted_sum += price * fill_size
                filled_size += fill_size
                remaining_size -= fill_size
                levels_consumed += 1
            
            if filled_size == 0:
                return {
                    'expected_average_price': 0.0,
                    'price_impact': float('inf'),
                    'levels_consumed': 0,
                    'unfilled_size': size,
                    'is_executable': False
                }
            
            average_price = weighted_sum / filled_size
            price_impact = abs(average_price - best_price) / best_price * 10000  # Convert to basis points
            
            return {
                'expected_average_price': average_price,
                'price_impact': price_impact,
                'levels_consumed': levels_consumed,
                'unfilled_size': remaining_size,
                'is_executable': price_impact <= self.max_slippage_bps
            }
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return {
                'expected_average_price': 0.0,
                'price_impact': float('inf'),
                'levels_consumed': 0,
                'unfilled_size': size,
                'is_executable': False
            }
    
    def get_execution_levels(
        self,
        order_book: Dict,
        side: str,
        size: float
    ) -> List[Dict]:
        """Break down how an order will be filled."""
        try:
            levels = []
            remaining_size = size
            running_total = 0.0
            running_size = 0.0
            
            book_levels = order_book['asks'] if side == 'buy' else order_book['bids']
            
            for level in book_levels:
                if remaining_size <= 0:
                    break
                
                price = float(level[0])
                available = float(level[1])
                
                fill_size = min(remaining_size, available)
                running_size += fill_size
                running_total += price * fill_size
                
                levels.append({
                    'price': price,
                    'size': fill_size,
                    'running_average': running_total / running_size
                })
                
                remaining_size -= fill_size
            
            return levels
            
        except Exception as e:
            logger.error(f"Error getting execution levels: {e}")
            return []
    
    def recommend_order_splits(
        self,
        order_book: Dict,
        side: str,
        size: float,
        max_slippage: float
    ) -> List[Dict]:
        """Recommend how to split a large order."""
        try:
            splits = []
            remaining_size = size
            delay = 0
            
            while remaining_size > 0:
                # Calculate slippage for current remaining size
                slippage = self.calculate_slippage(order_book, side, remaining_size)
                
                if slippage['price_impact'] <= max_slippage:
                    # Can execute remaining size in one order
                    splits.append({
                        'size': remaining_size,
                        'expected_price': slippage['expected_average_price'],
                        'expected_slippage': slippage['price_impact'],
                        'delay_seconds': delay
                    })
                    break
                
                # Try half the size
                test_size = remaining_size / 2
                while test_size > 0:
                    slippage = self.calculate_slippage(order_book, side, test_size)
                    if slippage['price_impact'] <= max_slippage:
                        splits.append({
                            'size': test_size,
                            'expected_price': slippage['expected_average_price'],
                            'expected_slippage': slippage['price_impact'],
                            'delay_seconds': delay
                        })
                        remaining_size -= test_size
                        delay += 30  # Add 30 second delay between splits
                        break
                    test_size /= 2
                
                if test_size <= 0:
                    # Cannot split further
                    logger.warning("Cannot split order to meet slippage requirements")
                    break
            
            return splits
            
        except Exception as e:
            logger.error(f"Error recommending order splits: {e}")
            return [] 