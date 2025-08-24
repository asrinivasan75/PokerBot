import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
from advanced_poker_env import AdvancedPokerEnv, PokerCard, Player
from neural_networks import create_poker_model

class BankrollManager:
    """Advanced bankroll management system"""
    
    def __init__(self, initial_bankroll: float = 10000.0, risk_tolerance: float = 0.02):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.risk_tolerance = risk_tolerance  # Risk per session as % of bankroll
        self.session_history = []
        self.buy_in_recommendations = {}
        
    def get_recommended_buy_in(self, stake_level: str) -> float:
        """Get recommended buy-in based on bankroll and stake level"""
        stake_multipliers = {
            "micro": 0.05,    # 5% of bankroll for micro stakes
            "low": 0.03,      # 3% for low stakes
            "medium": 0.02,   # 2% for medium stakes
            "high": 0.01,     # 1% for high stakes
            "nosebleed": 0.005 # 0.5% for highest stakes
        }
        
        multiplier = stake_multipliers.get(stake_level, 0.02)
        return self.current_bankroll * multiplier
    
    def can_afford_stake(self, buy_in: float, stake_level: str) -> bool:
        """Check if player can afford the stake level"""
        recommended = self.get_recommended_buy_in(stake_level)
        return buy_in <= recommended and self.current_bankroll >= buy_in * 20  # 20 buy-ins rule
    
    def record_session(self, profit_loss: float, duration_hours: float, stake_level: str):
        """Record a poker session"""
        session = {
            'profit_loss': profit_loss,
            'duration': duration_hours,
            'stake_level': stake_level,
            'roi': profit_loss / self.current_bankroll * 100,
            'hourly_rate': profit_loss / duration_hours if duration_hours > 0 else 0
        }
        
        self.session_history.append(session)
        self.current_bankroll += profit_loss
    
    def get_statistics(self) -> Dict:
        """Get bankroll statistics"""
        if not self.session_history:
            return {}
        
        total_profit = sum(s['profit_loss'] for s in self.session_history)
        total_hours = sum(s['duration'] for s in self.session_history)
        winning_sessions = len([s for s in self.session_history if s['profit_loss'] > 0])
        
        return {
            'current_bankroll': self.current_bankroll,
            'total_profit': total_profit,
            'roi': total_profit / self.initial_bankroll * 100,
            'hourly_rate': total_profit / total_hours if total_hours > 0 else 0,
            'win_rate': winning_sessions / len(self.session_history) * 100,
            'sessions_played': len(self.session_history),
            'biggest_win': max(s['profit_loss'] for s in self.session_history),
            'biggest_loss': min(s['profit_loss'] for s in self.session_history),
        }

class PotOddsCalculator:
    """Advanced pot odds and equity calculations"""
    
    @staticmethod
    def calculate_pot_odds(pot_size: float, bet_to_call: float) -> float:
        """Calculate pot odds"""
        if bet_to_call <= 0:
            return float('inf')
        return pot_size / (pot_size + bet_to_call)
    
    @staticmethod
    def calculate_implied_odds(pot_size: float, bet_to_call: float, 
                             future_bets: float, win_probability: float) -> float:
        """Calculate implied odds considering future betting"""
        total_pot = pot_size + future_bets
        return (total_pot * win_probability) / bet_to_call
    
    @staticmethod
    def calculate_reverse_implied_odds(pot_size: float, bet_to_call: float,
                                     potential_loss: float, lose_probability: float) -> float:
        """Calculate reverse implied odds (potential future losses)"""
        expected_loss = potential_loss * lose_probability
        return (pot_size - expected_loss) / bet_to_call
    
    @staticmethod
    def get_equity_vs_range(hole_cards: List[PokerCard], community_cards: List[PokerCard],
                           opponent_range: List[str]) -> float:
        """Calculate equity against opponent's range (simplified)"""
        # This would typically use a poker equity calculator
        # For now, return a simplified calculation based on hand strength
        hand_strength = PotOddsCalculator._estimate_hand_strength(hole_cards, community_cards)
        
        # Adjust based on opponent range tightness
        range_factor = len(opponent_range) / 1326  # Total possible hands
        equity = hand_strength * (1 - range_factor * 0.5)
        
        return max(0.05, min(0.95, equity))
    
    @staticmethod
    def _estimate_hand_strength(hole_cards: List[PokerCard], 
                               community_cards: List[PokerCard]) -> float:
        """Simplified hand strength estimation"""
        # This is a simplified version - real implementation would use proper evaluator
        if not hole_cards:
            return 0.0
        
        # Basic hand evaluation
        ranks = [card.rank for card in hole_cards + community_cards]
        suits = [card.suit for card in hole_cards + community_cards]
        
        # Check for pairs, straights, flushes, etc.
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        # Simple scoring
        if max_rank_count >= 4:
            return 0.95  # Four of a kind
        elif max_suit_count >= 5:
            return 0.85  # Flush
        elif max_rank_count >= 3:
            return 0.75  # Three of a kind
        elif max_rank_count >= 2:
            return 0.60  # Pair
        else:
            # High card - based on highest card
            high_card_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
            highest = max([high_card_values.get(rank, int(rank)) for rank in ranks])
            return 0.3 + (highest - 2) / 12 * 0.2

class BluffingStrategy:
    """Advanced bluffing and deception strategies"""
    
    def __init__(self):
        self.bluff_frequency = 0.15  # Baseline bluff frequency
        self.opponent_models = {}    # Track opponent tendencies
        self.board_texture_weights = {
            'dry': 1.2,      # Dry boards favor bluffing
            'wet': 0.8,      # Wet boards discourage bluffing
            'coordinated': 0.9,
            'rainbow': 1.1
        }
    
    def should_bluff(self, game_state: Dict, player_id: str) -> Tuple[bool, float]:
        """Determine if player should bluff and with what frequency"""
        
        # Factors affecting bluff decision
        factors = self._analyze_bluff_factors(game_state, player_id)
        
        # Calculate bluff probability
        base_frequency = self.bluff_frequency
        adjusted_frequency = base_frequency
        
        # Adjust based on factors
        for factor, weight in factors.items():
            if factor == 'position':
                adjusted_frequency *= weight
            elif factor == 'board_texture':
                adjusted_frequency *= weight
            elif factor == 'opponent_tendency':
                adjusted_frequency *= weight
            elif factor == 'stack_size':
                adjusted_frequency *= weight
        
        # Cap between 5% and 40%
        adjusted_frequency = max(0.05, min(0.40, adjusted_frequency))
        
        should_bluff = random.random() < adjusted_frequency
        return should_bluff, adjusted_frequency
    
    def _analyze_bluff_factors(self, game_state: Dict, player_id: str) -> Dict[str, float]:
        """Analyze factors that affect bluffing decision"""
        factors = {}
        
        # Position factor
        current_position = game_state.get('current_player', 0)
        total_players = len(game_state.get('players', []))
        
        if current_position >= total_players * 0.7:  # Late position
            factors['position'] = 1.3
        elif current_position <= total_players * 0.3:  # Early position
            factors['position'] = 0.7
        else:  # Middle position
            factors['position'] = 1.0
        
        # Board texture factor
        community_cards = game_state.get('community_cards', [])
        if len(community_cards) >= 3:
            texture = self._analyze_board_texture(community_cards)
            factors['board_texture'] = self.board_texture_weights.get(texture, 1.0)
        else:
            factors['board_texture'] = 1.0
        
        # Opponent tendency factor
        active_opponents = [p for p in game_state.get('players', []) 
                          if not p.get('folded', False) and p['id'] != player_id]
        
        if len(active_opponents) == 1:  # Heads up
            factors['opponent_tendency'] = 1.4
        elif len(active_opponents) > 3:  # Multiple opponents
            factors['opponent_tendency'] = 0.6
        else:
            factors['opponent_tendency'] = 1.0
        
        # Stack size factor
        player_stack = game_state.get('player_stack', 1000)
        big_blind = game_state.get('big_blind', 10)
        stack_bb = player_stack / big_blind
        
        if stack_bb > 100:  # Deep stacks
            factors['stack_size'] = 1.2
        elif stack_bb < 20:  # Short stacks
            factors['stack_size'] = 0.8
        else:
            factors['stack_size'] = 1.0
        
        return factors
    
    def _analyze_board_texture(self, community_cards: List[Dict]) -> str:
        """Analyze board texture for bluffing considerations"""
        if len(community_cards) < 3:
            return 'unknown'
        
        ranks = [card['rank'] for card in community_cards]
        suits = [card['suit'] for card in community_cards]
        
        # Check for coordinated boards
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        max_suit_count = max(suit_counts.values())
        
        # Check for connected ranks
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10}
        numeric_ranks = [rank_values.get(rank, int(rank)) for rank in ranks]
        numeric_ranks.sort()
        
        connected = all(numeric_ranks[i+1] - numeric_ranks[i] <= 2 
                       for i in range(len(numeric_ranks)-1))
        
        if max_suit_count >= 3 or connected:
            return 'wet'
        elif max_suit_count == len(set(suits)) and not connected:
            return 'dry'
        elif max_suit_count >= 2:
            return 'coordinated'
        else:
            return 'rainbow'
    
    def get_bluff_sizing(self, pot_size: float, position: str, board_texture: str) -> float:
        """Get recommended bluff bet sizing"""
        base_sizing = 0.6  # 60% of pot as base
        
        # Adjust based on position
        if position == 'late':
            sizing_multiplier = 1.1
        elif position == 'early':
            sizing_multiplier = 0.9
        else:
            sizing_multiplier = 1.0
        
        # Adjust based on board texture
        texture_multipliers = {
            'dry': 0.8,      # Smaller bluffs on dry boards
            'wet': 1.2,      # Larger bluffs on wet boards
            'coordinated': 1.1,
            'rainbow': 0.9
        }
        
        texture_multiplier = texture_multipliers.get(board_texture, 1.0)
        
        final_sizing = base_sizing * sizing_multiplier * texture_multiplier
        return pot_size * final_sizing

class TournamentManager:
    """Tournament structure and management"""
    
    def __init__(self, tournament_type: str = "freezeout"):
        self.tournament_type = tournament_type
        self.blind_levels = self._create_blind_structure()
        self.current_level = 0
        self.players = {}
        self.prize_pool = 0
        self.buy_in = 0
        self.start_time = None
        
    def _create_blind_structure(self) -> List[Dict]:
        """Create tournament blind structure"""
        if self.tournament_type == "turbo":
            # Faster blind increases
            return [
                {'level': 1, 'small_blind': 10, 'big_blind': 20, 'ante': 0, 'duration': 8},
                {'level': 2, 'small_blind': 15, 'big_blind': 30, 'ante': 0, 'duration': 8},
                {'level': 3, 'small_blind': 25, 'big_blind': 50, 'ante': 0, 'duration': 8},
                {'level': 4, 'small_blind': 50, 'big_blind': 100, 'ante': 10, 'duration': 8},
                {'level': 5, 'small_blind': 75, 'big_blind': 150, 'ante': 15, 'duration': 8},
                {'level': 6, 'small_blind': 100, 'big_blind': 200, 'ante': 20, 'duration': 8},
                {'level': 7, 'small_blind': 150, 'big_blind': 300, 'ante': 30, 'duration': 8},
                {'level': 8, 'small_blind': 200, 'big_blind': 400, 'ante': 40, 'duration': 8},
            ]
        else:  # Standard structure
            return [
                {'level': 1, 'small_blind': 10, 'big_blind': 20, 'ante': 0, 'duration': 15},
                {'level': 2, 'small_blind': 15, 'big_blind': 30, 'ante': 0, 'duration': 15},
                {'level': 3, 'small_blind': 25, 'big_blind': 50, 'ante': 0, 'duration': 15},
                {'level': 4, 'small_blind': 50, 'big_blind': 100, 'ante': 10, 'duration': 15},
                {'level': 5, 'small_blind': 75, 'big_blind': 150, 'ante': 15, 'duration': 15},
                {'level': 6, 'small_blind': 100, 'big_blind': 200, 'ante': 20, 'duration': 15},
                {'level': 7, 'small_blind': 150, 'big_blind': 300, 'ante': 30, 'duration': 15},
                {'level': 8, 'small_blind': 200, 'big_blind': 400, 'ante': 40, 'duration': 15},
            ]
    
    def get_current_blinds(self) -> Dict:
        """Get current blind level information"""
        if self.current_level < len(self.blind_levels):
            return self.blind_levels[self.current_level]
        else:
            # Double blinds if we exceed structure
            last_level = self.blind_levels[-1]
            multiplier = 2 ** (self.current_level - len(self.blind_levels) + 1)
            return {
                'level': self.current_level + 1,
                'small_blind': last_level['small_blind'] * multiplier,
                'big_blind': last_level['big_blind'] * multiplier,
                'ante': last_level['ante'] * multiplier,
                'duration': last_level['duration']
            }
    
    def advance_blind_level(self):
        """Advance to next blind level"""
        self.current_level += 1
    
    def calculate_payout_structure(self, num_players: int, buy_in: float) -> Dict[int, float]:
        """Calculate tournament payout structure"""
        total_prize_pool = num_players * buy_in
        
        if num_players <= 10:
            # Winner takes all for small tournaments
            return {1: total_prize_pool}
        elif num_players <= 50:
            # Top 3 get paid
            return {
                1: total_prize_pool * 0.5,
                2: total_prize_pool * 0.3,
                3: total_prize_pool * 0.2
            }
        else:
            # Top 10% get paid
            payouts = {}
            paid_positions = max(3, num_players // 10)
            
            # Standard payout structure
            payout_percentages = [0.3, 0.18, 0.12, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
            
            for position in range(1, paid_positions + 1):
                if position <= len(payout_percentages):
                    payouts[position] = total_prize_pool * payout_percentages[position - 1]
                else:
                    # Equal share for remaining positions
                    remaining_pool = total_prize_pool * 0.1
                    remaining_positions = paid_positions - len(payout_percentages)
                    payouts[position] = remaining_pool / remaining_positions
            
            return payouts

class SessionAnalyzer:
    """Analyze poker sessions and provide insights"""
    
    def __init__(self):
        self.session_data = []
        
    def add_hand(self, hand_data: Dict):
        """Add hand data for analysis"""
        self.session_data.append(hand_data)
    
    def analyze_session(self) -> Dict:
        """Analyze complete session"""
        if not self.session_data:
            return {}
        
        analysis = {
            'hands_played': len(self.session_data),
            'vpip': self._calculate_vpip(),
            'pfr': self._calculate_pfr(),
            'aggression_factor': self._calculate_aggression_factor(),
            'c_bet_frequency': self._calculate_cbet_frequency(),
            'fold_to_cbet': self._calculate_fold_to_cbet(),
            'showdown_winnings': self._calculate_showdown_winnings(),
            'non_showdown_winnings': self._calculate_non_showdown_winnings(),
            'positional_stats': self._analyze_positional_play(),
            'hand_strength_analysis': self._analyze_hand_strength_play()
        }
        
        return analysis
    
    def _calculate_vpip(self) -> float:
        """Calculate Voluntarily Put In Pot percentage"""
        vpip_hands = sum(1 for hand in self.session_data 
                        if hand.get('voluntary_investment', False))
        return vpip_hands / len(self.session_data) * 100
    
    def _calculate_pfr(self) -> float:
        """Calculate Pre-Flop Raise percentage"""
        pfr_hands = sum(1 for hand in self.session_data 
                       if hand.get('preflop_raise', False))
        return pfr_hands / len(self.session_data) * 100
    
    def _calculate_aggression_factor(self) -> float:
        """Calculate aggression factor (bets + raises) / calls"""
        aggressive_actions = sum(hand.get('bets', 0) + hand.get('raises', 0) 
                               for hand in self.session_data)
        calls = sum(hand.get('calls', 0) for hand in self.session_data)
        
        return aggressive_actions / calls if calls > 0 else 0
    
    def _calculate_cbet_frequency(self) -> float:
        """Calculate continuation bet frequency"""
        cbet_opportunities = sum(1 for hand in self.session_data 
                               if hand.get('cbet_opportunity', False))
        cbets_made = sum(1 for hand in self.session_data 
                        if hand.get('cbet_made', False))
        
        return cbets_made / cbet_opportunities * 100 if cbet_opportunities > 0 else 0
    
    def _calculate_fold_to_cbet(self) -> float:
        """Calculate fold to continuation bet frequency"""
        faced_cbet = sum(1 for hand in self.session_data 
                        if hand.get('faced_cbet', False))
        folded_to_cbet = sum(1 for hand in self.session_data 
                           if hand.get('folded_to_cbet', False))
        
        return folded_to_cbet / faced_cbet * 100 if faced_cbet > 0 else 0
    
    def _calculate_showdown_winnings(self) -> float:
        """Calculate showdown winnings"""
        showdown_hands = [hand for hand in self.session_data 
                         if hand.get('went_to_showdown', False)]
        
        if not showdown_hands:
            return 0
        
        total_winnings = sum(hand.get('winnings', 0) for hand in showdown_hands)
        return total_winnings
    
    def _calculate_non_showdown_winnings(self) -> float:
        """Calculate non-showdown winnings"""
        non_showdown_hands = [hand for hand in self.session_data 
                             if not hand.get('went_to_showdown', False)]
        
        if not non_showdown_hands:
            return 0
        
        total_winnings = sum(hand.get('winnings', 0) for hand in non_showdown_hands)
        return total_winnings
    
    def _analyze_positional_play(self) -> Dict:
        """Analyze play by position"""
        positions = ['early', 'middle', 'late', 'blinds']
        positional_stats = {}
        
        for position in positions:
            position_hands = [hand for hand in self.session_data 
                            if hand.get('position') == position]
            
            if position_hands:
                positional_stats[position] = {
                    'hands': len(position_hands),
                    'vpip': sum(1 for hand in position_hands 
                              if hand.get('voluntary_investment', False)) / len(position_hands) * 100,
                    'pfr': sum(1 for hand in position_hands 
                             if hand.get('preflop_raise', False)) / len(position_hands) * 100,
                    'winnings': sum(hand.get('winnings', 0) for hand in position_hands)
                }
        
        return positional_stats
    
    def _analyze_hand_strength_play(self) -> Dict:
        """Analyze play based on hand strength"""
        strength_categories = ['premium', 'strong', 'medium', 'weak']
        strength_stats = {}
        
        for category in strength_categories:
            category_hands = [hand for hand in self.session_data 
                            if hand.get('hand_strength_category') == category]
            
            if category_hands:
                strength_stats[category] = {
                    'hands': len(category_hands),
                    'played_rate': sum(1 for hand in category_hands 
                                     if hand.get('voluntary_investment', False)) / len(category_hands) * 100,
                    'win_rate': sum(1 for hand in category_hands 
                                  if hand.get('winnings', 0) > 0) / len(category_hands) * 100,
                    'avg_winnings': np.mean([hand.get('winnings', 0) for hand in category_hands])
                }
        
        return strength_stats

# Integration class that combines all advanced features
class AdvancedPokerBot:
    """Advanced poker bot with all features integrated"""
    
    def __init__(self, model_type: str = "transformer"):
        self.model = create_poker_model(model_type)
        self.bankroll_manager = BankrollManager()
        self.pot_odds_calculator = PotOddsCalculator()
        self.bluffing_strategy = BluffingStrategy()
        self.session_analyzer = SessionAnalyzer()
        self.tournament_manager = None
        
    def make_decision(self, game_state: Dict, player_id: str) -> Tuple[int, Dict]:
        """Make a poker decision using all advanced features"""
        
        # Calculate pot odds
        pot_size = game_state.get('pot', 0)
        current_bet = game_state.get('current_bet', 0)
        player_bet = game_state.get('player_current_bet', 0)
        call_amount = current_bet - player_bet
        
        pot_odds = self.pot_odds_calculator.calculate_pot_odds(pot_size, call_amount)
        
        # Get hand strength and equity
        hole_cards = game_state.get('player_cards', [])
        community_cards = game_state.get('community_cards', [])
        
        # Convert card dicts to PokerCard objects if needed
        if hole_cards and isinstance(hole_cards[0], dict):
            hole_cards = [PokerCard(card['rank'], card['suit']) for card in hole_cards]
        if community_cards and isinstance(community_cards[0], dict):
            community_cards = [PokerCard(card['rank'], card['suit']) for card in community_cards]
        
        hand_strength = self.pot_odds_calculator._estimate_hand_strength(hole_cards, community_cards)
        
        # Check if we should bluff
        should_bluff, bluff_frequency = self.bluffing_strategy.should_bluff(game_state, player_id)
        
        # Get neural network recommendation
        valid_actions = game_state.get('valid_actions', [])
        
        if valid_actions:
            try:
                # Create observation for neural network
                observation = self._create_observation(game_state)
                policy, value = self.model.predict(observation.reshape(1, -1), verbose=0)
                
                # Mask invalid actions
                masked_policy = policy[0].copy()
                for i in range(len(masked_policy)):
                    if i not in valid_actions:
                        masked_policy[i] = 0
                
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                    
                    # Adjust probabilities based on advanced features
                    adjusted_policy = self._adjust_policy(
                        masked_policy, pot_odds, hand_strength, should_bluff, game_state
                    )
                    
                    action = np.random.choice(len(adjusted_policy), p=adjusted_policy)
                else:
                    action = random.choice(valid_actions)
                    
            except Exception as e:
                print(f"Error in neural network decision: {e}")
                action = self._fallback_decision(game_state, pot_odds, hand_strength)
        else:
            action = 0  # Fold if no valid actions
        
        # Create decision explanation
        decision_info = {
            'action': action,
            'pot_odds': pot_odds,
            'hand_strength': hand_strength,
            'should_bluff': should_bluff,
            'bluff_frequency': bluff_frequency,
            'call_amount': call_amount,
            'reasoning': self._generate_reasoning(action, pot_odds, hand_strength, should_bluff)
        }
        
        return action, decision_info
    
    def _create_observation(self, game_state: Dict) -> np.ndarray:
        """Create observation vector for neural network"""
        # This is a simplified version - would need to match the actual environment's observation space
        observation = np.zeros(428)  # Match AdvancedPokerEnv observation space
        
        # Fill with available game state information
        # This would need to be implemented to match the actual environment's encoding
        
        return observation
    
    def _adjust_policy(self, policy: np.ndarray, pot_odds: float, hand_strength: float, 
                      should_bluff: bool, game_state: Dict) -> np.ndarray:
        """Adjust neural network policy based on advanced features"""
        adjusted = policy.copy()
        
        # If we have a strong hand, increase betting probabilities
        if hand_strength > 0.7:
            # Increase betting actions (3-7)
            for i in range(3, 8):
                if i < len(adjusted):
                    adjusted[i] *= 1.5
        
        # If pot odds are favorable, increase calling probability
        if pot_odds > 0.3 and hand_strength > 0.4:
            if 2 < len(adjusted):  # Call action
                adjusted[2] *= 1.3
        
        # If we should bluff, increase betting probabilities for weak hands
        if should_bluff and hand_strength < 0.3:
            for i in range(3, 8):
                if i < len(adjusted):
                    adjusted[i] *= 1.8
            # Decrease calling and checking
            if len(adjusted) > 2:
                adjusted[1] *= 0.5  # Check
                adjusted[2] *= 0.3  # Call
        
        # Renormalize
        if adjusted.sum() > 0:
            adjusted /= adjusted.sum()
        
        return adjusted
    
    def _fallback_decision(self, game_state: Dict, pot_odds: float, hand_strength: float) -> int:
        """Fallback decision logic when neural network fails"""
        call_amount = game_state.get('current_bet', 0) - game_state.get('player_current_bet', 0)
        
        if call_amount == 0:
            return 1  # Check
        elif hand_strength > 0.7:
            return 6  # Bet 100%
        elif hand_strength > 0.5 and pot_odds > 0.3:
            return 2  # Call
        elif hand_strength > 0.3 and pot_odds > 0.4:
            return 2  # Call
        else:
            return 0  # Fold
    
    def _generate_reasoning(self, action: int, pot_odds: float, hand_strength: float, 
                          should_bluff: bool) -> str:
        """Generate human-readable reasoning for the decision"""
        action_names = {
            0: "Fold", 1: "Check", 2: "Call", 3: "Bet 25%", 
            4: "Bet 50%", 5: "Bet 75%", 6: "Bet 100%", 7: "All In"
        }
        
        action_name = action_names.get(action, f"Action {action}")
        
        if action == 0:
            return f"Folding due to weak hand (strength: {hand_strength:.2f}) and unfavorable pot odds ({pot_odds:.2f})"
        elif action in [1, 2]:
            return f"Playing cautiously with moderate hand strength ({hand_strength:.2f}) and pot odds ({pot_odds:.2f})"
        elif action in range(3, 8):
            if should_bluff:
                return f"Bluffing with {action_name} based on position and board texture"
            else:
                return f"Value betting with {action_name} due to strong hand (strength: {hand_strength:.2f})"
        
        return f"Chose {action_name}"

if __name__ == "__main__":
    # Example usage of advanced features
    
    # Bankroll management example
    bankroll = BankrollManager(initial_bankroll=5000)
    print(f"Recommended buy-in for medium stakes: ${bankroll.get_recommended_buy_in('medium'):.2f}")
    
    # Record some sessions
    bankroll.record_session(150, 2.5, "low")
    bankroll.record_session(-75, 1.8, "low")
    bankroll.record_session(300, 4.0, "medium")
    
    print("Bankroll statistics:", bankroll.get_statistics())
    
    # Pot odds calculation example
    pot_odds = PotOddsCalculator.calculate_pot_odds(100, 25)
    print(f"Pot odds: {pot_odds:.2f}")
    
    # Tournament example
    tournament = TournamentManager("turbo")
    current_blinds = tournament.get_current_blinds()
    print(f"Current blinds: {current_blinds}")
    
    payouts = tournament.calculate_payout_structure(100, 50)
    print(f"Payout structure for 100 players: {payouts}")
    
    print("Advanced poker features implemented successfully!")