from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class OrderBookLevel:
    price: Decimal
    size: Decimal
    
class OrderBookAnalyzer:
    """
    Handles detailed order book analysis and execution price calculations.
    """
    
    def __init__(
        self,
        min_spread_threshold: float = 0.0001,  # 0.01%
        max_spread_threshold: float = 0.005,   # 0.5%
        min_depth_threshold: float = 10000,    # $10,000 equivalent
        max_price_impact: float = 0.003        # 0.3%
    ):
        """
        Initialize the OrderBookAnalyzer.
        
        Args:
            min_spread_threshold: Minimum acceptable spread (as decimal)
            max_spread_threshold: Maximum acceptable spread (as decimal)
            min_depth_threshold: Minimum required depth in quote currency
            max_price_impact: Maximum acceptable price impact (as decimal)
        """
        self.min_spread_threshold = min_spread_threshold
        self.max_spread_threshold = max_spread_threshold
        self.min_depth_threshold = min_depth_threshold
        self.max_price_impact = max_price_impact
    
    def calculate_spread(self, order_book: Dict) -> Dict:
        """
        Calculate current spread metrics.
        
        Args:
            order_book: Dictionary containing bids and asks
            
        Returns:
            Dict containing spread metrics
        """
        try:
            best_bid = Decimal(str(order_book['bids'][0][0]))
            best_ask = Decimal(str(order_book['asks'][0][0]))
            bid_size = Decimal(str(order_book['bids'][0][1]))
            ask_size = Decimal(str(order_book['asks'][0][1]))
            
            absolute_spread = best_ask - best_bid
            relative_spread = absolute_spread / best_bid
            
            is_tradeable = (
                relative_spread >= self.min_spread_threshold and
                relative_spread <= self.max_spread_threshold and
                min(bid_size, ask_size) * best_bid >= self.min_depth_threshold
            )
            
            return {
                'absolute_spread': float(absolute_spread),
                'relative_spread': float(relative_spread),
                'bid_depth': float(bid_size),
                'ask_depth': float(ask_size),
                'is_tradeable': is_tradeable
            }
            
        except Exception as e:
            logger.error(f"Error calculating spread metrics: {e}")
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
        """
        Calculate expected slippage for an order.
        
        Args:
            order_book: Dictionary containing bids and asks
            side: 'buy' or 'sell'
            size: Order size in base currency
            
        Returns:
            Dict containing slippage analysis
        """
        try:
            # Convert order book levels to OrderBookLevel objects
            levels = []
            raw_levels = order_book['asks'] if side == 'buy' else order_book['bids']
            
            for price, amount in raw_levels:
                levels.append(OrderBookLevel(
                    price=Decimal(str(price)),
                    size=Decimal(str(amount))
                ))
            
            remaining_size = Decimal(str(size))
            total_cost = Decimal('0')
            levels_consumed = 0
            best_price = levels[0].price
            
            # Calculate weighted average price
            for level in levels:
                if remaining_size <= 0:
                    break
                    
                fill_size = min(remaining_size, level.size)
                total_cost += fill_size * level.price
                remaining_size -= fill_size
                levels_consumed += 1
            
            if remaining_size > 0:
                return {
                    'expected_average_price': float('inf'),
                    'price_impact': float('inf'),
                    'levels_consumed': levels_consumed,
                    'unfilled_size': float(remaining_size),
                    'is_executable': False
                }
            
            average_price = total_cost / Decimal(str(size))
            price_impact = (average_price - best_price) / best_price
            
            return {
                'expected_average_price': float(average_price),
                'price_impact': float(price_impact),
                'levels_consumed': levels_consumed,
                'unfilled_size': 0.0,
                'is_executable': price_impact <= self.max_price_impact
            }
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return {
                'expected_average_price': float('inf'),
                'price_impact': float('inf'),
                'levels_consumed': 0,
                'unfilled_size': float(size),
                'is_executable': False
            }
    
    def get_execution_levels(
        self,
        order_book: Dict,
        side: str,
        size: float
    ) -> List[Dict]:
        """
        Break down how an order will be filled across price levels.
        
        Args:
            order_book: Dictionary containing bids and asks
            side: 'buy' or 'sell'
            size: Order size in base currency
            
        Returns:
            List of dictionaries containing fill information per level
        """
        try:
            levels = []
            raw_levels = order_book['asks'] if side == 'buy' else order_book['bids']
            remaining_size = Decimal(str(size))
            total_filled = Decimal('0')
            total_cost = Decimal('0')
            
            for price, amount in raw_levels:
                if remaining_size <= 0:
                    break
                    
                price = Decimal(str(price))
                amount = Decimal(str(amount))
                fill_size = min(remaining_size, amount)
                
                total_filled += fill_size
                level_cost = fill_size * price
                total_cost += level_cost
                
                levels.append({
                    'price': float(price),
                    'size': float(fill_size),
                    'running_average': float(total_cost / total_filled)
                })
                
                remaining_size -= fill_size
            
            return levels
            
        except Exception as e:
            logger.error(f"Error analyzing execution levels: {e}")
            return []
    
    def recommend_order_splits(
        self,
        order_book: Dict,
        side: str,
        size: float,
        max_slippage: float
    ) -> List[Dict]:
        """
        Recommend how to split a large order to minimize market impact.
        
        Args:
            order_book: Dictionary containing bids and asks
            side: 'buy' or 'sell'
            size: Total order size in base currency
            max_slippage: Maximum acceptable slippage per split
            
        Returns:
            List of dictionaries containing split recommendations
        """
        try:
            # Start with analysis of full size
            full_analysis = self.calculate_slippage(order_book, side, size)
            
            # If can execute full size within slippage, return single order
            if full_analysis['price_impact'] <= max_slippage:
                return [{
                    'size': size,
                    'expected_price': full_analysis['expected_average_price'],
                    'expected_slippage': full_analysis['price_impact'],
                    'delay_seconds': 0
                }]
            
            # Otherwise, split the order
            splits = []
            remaining_size = size
            base_delay = 30  # Base delay between orders in seconds
            
            # Binary search for optimal split size
            test_size = size / 2
            min_size = 0
            max_size = size
            
            while max_size - min_size > size * 0.01:  # 1% precision
                analysis = self.calculate_slippage(order_book, side, float(test_size))
                
                if analysis['price_impact'] <= max_slippage:
                    min_size = test_size
                else:
                    max_size = test_size
                    
                test_size = (min_size + max_size) / 2
            
            optimal_split_size = min_size
            
            # Generate splits
            split_number = 0
            while remaining_size > 0:
                current_size = min(optimal_split_size, remaining_size)
                analysis = self.calculate_slippage(
                    order_book,
                    side,
                    float(current_size)
                )
                
                splits.append({
                    'size': float(current_size),
                    'expected_price': analysis['expected_average_price'],
                    'expected_slippage': analysis['price_impact'],
                    'delay_seconds': split_number * base_delay
                })
                
                remaining_size -= current_size
                split_number += 1
            
            return splits
            
        except Exception as e:
            logger.error(f"Error calculating order splits: {e}")
            return [{
                'size': size,
                'expected_price': float('inf'),
                'expected_slippage': float('inf'),
                'delay_seconds': 0
            }] 