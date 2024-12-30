from typing import Dict
from loguru import logger
from src.analysis.order_book_analyzer import OrderBookAnalyzer

class OrderExecutor:
    def __init__(
        self,
        price_feed,
        max_spread_bps: int = 10,        # Maximum allowed spread in basis points
        max_slippage_bps: int = 20,      # Maximum allowed slippage in basis points
        min_depth_usd: float = 100000,   # Minimum depth required at best bid/ask
        impact_threshold_bps: int = 30    # Max allowed market impact in basis points
    ):
        self.price_feed = price_feed
        self.order_book_analyzer = OrderBookAnalyzer(
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
            # Analyze spread
            spread_metrics = self.order_book_analyzer.calculate_spread(order_book)
            
            # Analyze slippage
            slippage_metrics = self.order_book_analyzer.calculate_slippage(
                order_book,
                side,
                size
            )
            
            # Get execution levels
            execution_levels = self.order_book_analyzer.get_execution_levels(
                order_book,
                side,
                size
            )
            
            # Check if conditions are acceptable
            spread_acceptable = spread_metrics['relative_spread'] <= self.order_book_analyzer.max_spread_bps
            slippage_acceptable = slippage_metrics['price_impact'] <= self.order_book_analyzer.max_slippage_bps
            depth_sufficient = (
                spread_metrics['bid_depth'] >= self.order_book_analyzer.min_depth_usd and
                spread_metrics['ask_depth'] >= self.order_book_analyzer.min_depth_usd
            )
            
            # Determine if order should be split
            should_split = (
                size > spread_metrics['bid_depth'] * 0.1 or
                size > spread_metrics['ask_depth'] * 0.1 or
                slippage_metrics['price_impact'] > self.order_book_analyzer.max_slippage_bps * 0.5
            )
            
            # Get recommended splits if needed
            recommended_splits = []
            if should_split:
                recommended_splits = self.order_book_analyzer.recommend_order_splits(
                    order_book,
                    side,
                    size,
                    self.order_book_analyzer.max_slippage_bps
                )
            
            return {
                'spread': {
                    'absolute': spread_metrics['absolute_spread'],
                    'relative_bps': spread_metrics['relative_spread'],
                    'is_acceptable': spread_acceptable
                },
                'slippage': {
                    'expected_bps': slippage_metrics['price_impact'],
                    'is_acceptable': slippage_acceptable
                },
                'depth': {
                    'available_size': min(spread_metrics['bid_depth'], spread_metrics['ask_depth']),
                    'is_sufficient': depth_sufficient
                },
                'recommendation': {
                    'should_execute': spread_acceptable and slippage_acceptable and depth_sufficient,
                    'split_orders': should_split,
                    'recommended_splits': recommended_splits
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing execution conditions: {str(e)}")
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
            best_price = levels[0]['price']
            weighted_price = sum(level['price'] * level['size'] for level in levels)
            total_size = sum(level['size'] for level in levels)
            average_price = weighted_price / total_size if total_size > 0 else 0.0
            
            # Calculate spread cost
            spread_metrics = self.order_book_analyzer.calculate_spread(order_book)
            spread_cost = spread_metrics['absolute_spread'] * size
            
            # Calculate slippage cost
            slippage_cost = (average_price - best_price) * size if side == 'buy' else (best_price - average_price) * size
            
            # Calculate total cost
            gross_cost = average_price * size
            total_cost = gross_cost + spread_cost + slippage_cost
            
            # Calculate metrics in basis points
            spread_bps = spread_metrics['relative_spread']
            slippage_bps = abs(average_price - best_price) / best_price * 10000
            total_cost_bps = spread_bps + slippage_bps
            
            return {
                'base_asset': {
                    'amount': size,
                    'average_price': average_price
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
            logger.error(f"Error estimating total cost: {str(e)}")
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
    
    async def execute_order(
        self,
        symbol: str,
        side: str,
        size: float,
        max_slippage_bps: float = None
    ) -> Dict:
        """Execute an order with smart order routing."""
        try:
            # Use instance max slippage if not specified
            if max_slippage_bps is None:
                max_slippage_bps = self.order_book_analyzer.max_slippage_bps
            
            # Get current order book
            order_book = await self.price_feed.get_order_book(symbol)
            
            # Analyze execution conditions
            conditions = await self.analyze_execution_conditions(
                symbol,
                side,
                size,
                order_book
            )
            
            if not conditions['recommendation']['should_execute']:
                return {
                    'success': False,
                    'reason': 'Execution conditions not met',
                    'details': conditions
                }
            
            if conditions['recommendation']['split_orders']:
                # Execute order in splits
                splits = conditions['recommendation']['recommended_splits']
                executed_splits = []
                remaining_size = size
                
                for split in splits:
                    if remaining_size <= 0:
                        break
                    
                    # Update order book before each split
                    order_book = await self.price_feed.get_order_book(symbol)
                    
                    # Execute split
                    result = await self._execute_single_order(
                        symbol,
                        side,
                        split['size'],
                        order_book,
                        max_slippage_bps
                    )
                    
                    if result['success']:
                        executed_splits.append(result)
                        remaining_size -= split['size']
                    else:
                        break
                
                return {
                    'success': len(executed_splits) > 0,
                    'executed_splits': executed_splits,
                    'remaining_size': remaining_size
                }
                
            else:
                # Execute order in one go
                return await self._execute_single_order(
                    symbol,
                    side,
                    size,
                    order_book,
                    max_slippage_bps
                )
            
        except Exception as e:
            logger.error(f"Error executing order: {str(e)}")
            return {
                'success': False,
                'reason': f'Execution error: {str(e)}'
            }
    
    async def _execute_single_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_book: Dict,
        max_slippage_bps: float
    ) -> Dict:
        """Execute a single order."""
        try:
            # Estimate execution cost
            cost_estimate = await self.estimate_total_cost(
                symbol,
                side,
                size,
                order_book
            )
            
            if cost_estimate['metrics']['total_cost_bps'] > max_slippage_bps:
                return {
                    'success': False,
                    'reason': 'Slippage too high',
                    'cost_estimate': cost_estimate
                }
            
            # Execute order (mock for now)
            execution_price = cost_estimate['base_asset']['average_price']
            
            return {
                'success': True,
                'executed_price': execution_price,
                'executed_size': size,
                'cost_metrics': cost_estimate['metrics']
            }
            
        except Exception as e:
            logger.error(f"Error executing single order: {str(e)}")
            return {
                'success': False,
                'reason': f'Single order execution error: {str(e)}'
            } 