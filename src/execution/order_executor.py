from typing import Dict, List, Optional
from loguru import logger
from ..analysis.order_book_analyzer import OrderBookAnalyzer

class OrderExecutor:
    """Handles intelligent order execution."""
    
    def __init__(
        self,
        price_feed,
        order_book_analyzer: Optional[OrderBookAnalyzer] = None,
        max_spread_bps: int = 10,        # Maximum allowed spread in basis points
        max_slippage_bps: int = 20,      # Maximum allowed slippage in basis points
        min_depth_usd: float = 100000,   # Minimum depth required at best bid/ask
        impact_threshold_bps: int = 30    # Max allowed market impact in basis points
    ):
        self.price_feed = price_feed
        self.order_book_analyzer = order_book_analyzer or OrderBookAnalyzer(
            max_spread_bps=max_spread_bps,
            max_slippage_bps=max_slippage_bps,
            min_depth_usd=min_depth_usd,
            impact_threshold_bps=impact_threshold_bps
        )
    
    async def analyze_execution_conditions(
        self,
        symbol: str,
        side: str,
        size: float,
        order_book: Dict
    ) -> Dict:
        """Comprehensive analysis of execution conditions."""
        try:
            # Analyze spread conditions
            spread_metrics = self.order_book_analyzer.calculate_spread(order_book)
            
            # Analyze slippage
            slippage_metrics = self.order_book_analyzer.calculate_slippage(
                order_book,
                side,
                size
            )
            
            # Check if order needs to be split
            splits = []
            if (not slippage_metrics['is_executable'] and 
                slippage_metrics['price_impact'] > self.order_book_analyzer.max_slippage_bps):
                splits = self.order_book_analyzer.recommend_order_splits(
                    order_book,
                    side,
                    size,
                    self.order_book_analyzer.max_slippage_bps
                )
            
            return {
                'spread': {
                    'absolute': spread_metrics['absolute_spread'],
                    'relative_bps': spread_metrics['relative_spread'],
                    'is_acceptable': spread_metrics['is_tradeable']
                },
                'slippage': {
                    'expected_bps': slippage_metrics['price_impact'],
                    'is_acceptable': slippage_metrics['is_executable']
                },
                'depth': {
                    'available_size': (
                        spread_metrics['bid_depth'] if side == 'sell'
                        else spread_metrics['ask_depth']
                    ),
                    'is_sufficient': spread_metrics['is_tradeable']
                },
                'recommendation': {
                    'should_execute': (
                        spread_metrics['is_tradeable'] and
                        slippage_metrics['is_executable']
                    ),
                    'split_orders': len(splits) > 0,
                    'recommended_splits': splits
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing execution conditions: {e}")
            return {
                'spread': {
                    'absolute': float('inf'),
                    'relative_bps': float('inf'),
                    'is_acceptable': False
                },
                'slippage': {
                    'expected_bps': float('inf'),
                    'is_acceptable': False
                },
                'depth': {
                    'available_size': 0.0,
                    'is_sufficient': False
                },
                'recommendation': {
                    'should_execute': False,
                    'split_orders': False,
                    'recommended_splits': []
                }
            }
    
    async def estimate_total_cost(
        self,
        symbol: str,
        side: str,
        size: float,
        order_book: Dict
    ) -> Dict:
        """Estimate total trading cost."""
        try:
            # Get execution levels
            levels = self.order_book_analyzer.get_execution_levels(
                order_book,
                side,
                size
            )
            
            if not levels:
                return {
                    'base_asset': {
                        'amount': 0.0,
                        'average_price': 0.0
                    },
                    'quote_asset': {
                        'gross_cost': 0.0,
                        'spread_cost': 0.0,
                        'slippage_cost': 0.0,
                        'total_cost': 0.0
                    },
                    'metrics': {
                        'spread_bps': 0.0,
                        'slippage_bps': 0.0,
                        'total_cost_bps': 0.0
                    }
                }
            
            # Calculate costs
            best_price = float(order_book['asks' if side == 'buy' else 'bids'][0][0])
            total_size = sum(level['size'] for level in levels)
            weighted_price = sum(level['price'] * level['size'] for level in levels) / total_size
            
            # Calculate spread cost
            spread_metrics = self.order_book_analyzer.calculate_spread(order_book)
            spread_cost = spread_metrics['absolute_spread'] * total_size / 2  # Half the spread
            
            # Calculate slippage cost
            slippage_cost = (weighted_price - best_price) * total_size if side == 'buy' else (best_price - weighted_price) * total_size
            
            # Total cost
            gross_cost = weighted_price * total_size
            total_cost = gross_cost + spread_cost + slippage_cost
            
            # Convert to basis points
            spread_bps = spread_metrics['relative_spread']
            slippage_bps = abs(weighted_price - best_price) / best_price * 10000
            total_cost_bps = total_cost / (best_price * total_size) * 10000
            
            return {
                'base_asset': {
                    'amount': total_size,
                    'average_price': weighted_price
                },
                'quote_asset': {
                    'gross_cost': gross_cost,
                    'spread_cost': spread_cost,
                    'slippage_cost': slippage_cost,
                    'total_cost': total_cost
                },
                'metrics': {
                    'spread_bps': spread_bps,
                    'slippage_bps': slippage_bps,
                    'total_cost_bps': total_cost_bps
                }
            }
            
        except Exception as e:
            logger.error(f"Error estimating total cost: {e}")
            return {
                'base_asset': {
                    'amount': 0.0,
                    'average_price': 0.0
                },
                'quote_asset': {
                    'gross_cost': 0.0,
                    'spread_cost': 0.0,
                    'slippage_cost': 0.0,
                    'total_cost': 0.0
                },
                'metrics': {
                    'spread_bps': 0.0,
                    'slippage_bps': 0.0,
                    'total_cost_bps': 0.0
                }
            } 