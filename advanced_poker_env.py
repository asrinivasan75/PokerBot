import numpy as np
import gym
from gym import spaces
import random
from typing import List, Dict, Tuple, Optional
from enum import Enum
from treys import Evaluator, Card
import json

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET_25 = 3
    BET_50 = 4
    BET_75 = 5
    BET_100 = 6
    ALL_IN = 7

class GamePhase(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

class Position(Enum):
    SMALL_BLIND = 0
    BIG_BLIND = 1
    EARLY = 2
    MIDDLE = 3
    LATE = 4
    BUTTON = 5

class PokerCard:
    SUITS = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    RANKS = {'2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', 
             '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A'}
    
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        
    def __repr__(self):
        return f"{PokerCard.RANKS[self.rank]}{PokerCard.SUITS[self.suit]}"
    
    def to_treys(self):
        return Card.new(f"{self.rank}{self.suit}")

class Player:
    def __init__(self, name: str, stack: int = 1000, position: Position = Position.BUTTON):
        self.name = name
        self.stack = stack
        self.hole_cards = []
        self.position = position
        self.current_bet = 0
        self.folded = False
        self.all_in = False
        self.actions_this_round = []
        
    def reset_for_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.folded = False
        self.all_in = False
        self.actions_this_round = []

class AdvancedPokerEnv(gym.Env):
    def __init__(self, num_players: int = 6, starting_stack: int = 1000, 
                 small_blind: int = 5, big_blind: int = 10):
        super().__init__()
        
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Action space: 8 possible actions
        self.action_space = spaces.Discrete(8)
        
        # Enhanced observation space: 
        # - Hand cards (104 features - one-hot encoded)
        # - Community cards (260 features - one-hot encoded)
        # - Betting history (50 features)
        # - Stack sizes (6 features)
        # - Position info (6 features)
        # - Pot odds (1 feature)
        # - Hand strength (1 feature)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(428,), dtype=np.float32
        )
        
        self.evaluator = Evaluator()
        self.players = []
        self.current_player = 0
        self.dealer_button = 0
        self.deck = []
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.game_phase = GamePhase.PREFLOP
        self.betting_round_actions = []
        
        self._initialize_players()
        
    def _initialize_players(self):
        positions = list(Position)
        for i in range(self.num_players):
            pos = positions[i % len(positions)]
            self.players.append(Player(f"Player_{i}", self.starting_stack, pos))
    
    def _create_deck(self):
        ranks = '23456789TJQKA'
        suits = 'shdc'
        self.deck = [PokerCard(rank, suit) for rank in ranks for suit in suits]
        random.shuffle(self.deck)
    
    def _deal_hole_cards(self):
        for player in self.players:
            if not player.folded:
                player.hole_cards = [self.deck.pop(), self.deck.pop()]
    
    def _post_blinds(self):
        sb_player = (self.dealer_button + 1) % self.num_players
        bb_player = (self.dealer_button + 2) % self.num_players
        
        self.players[sb_player].current_bet = self.small_blind
        self.players[sb_player].stack -= self.small_blind
        self.pot += self.small_blind
        
        self.players[bb_player].current_bet = self.big_blind
        self.players[bb_player].stack -= self.big_blind
        self.pot += self.big_blind
        
        self.current_bet = self.big_blind
    
    def _encode_cards(self, cards: List[PokerCard], max_cards: int) -> np.ndarray:
        """One-hot encode cards"""
        encoding = np.zeros(max_cards * 52)
        for i, card in enumerate(cards[:max_cards]):
            if card:
                rank_idx = '23456789TJQKA'.index(card.rank)
                suit_idx = 'shdc'.index(card.suit)
                card_idx = rank_idx * 4 + suit_idx
                encoding[i * 52 + card_idx] = 1
        return encoding
    
    def _get_hand_strength(self, player: Player) -> float:
        if not player.hole_cards or len(self.community_cards) < 3:
            return 0.0
            
        try:
            hole_cards_treys = [card.to_treys() for card in player.hole_cards]
            community_treys = [card.to_treys() for card in self.community_cards]
            
            if len(community_treys) >= 3:
                rank = self.evaluator.evaluate(community_treys, hole_cards_treys)
                return 1 - (rank / 7462)  # Normalize to 0-1
        except:
            return 0.0
        return 0.0
    
    def _get_pot_odds(self, player: Player) -> float:
        call_amount = self.current_bet - player.current_bet
        if call_amount <= 0:
            return 1.0
        return self.pot / (self.pot + call_amount)
    
    def _get_observation(self, player_idx: int) -> np.ndarray:
        player = self.players[player_idx]
        
        # Encode hole cards (2 cards * 52 = 104 features)
        hole_encoding = self._encode_cards(player.hole_cards, 2)
        
        # Encode community cards (5 cards * 52 = 260 features)
        community_encoding = self._encode_cards(self.community_cards, 5)
        
        # Betting history (last 50 actions)
        betting_history = np.zeros(50)
        recent_actions = self.betting_round_actions[-50:]
        for i, action in enumerate(recent_actions):
            betting_history[i] = action / 7  # Normalize action values
            
        # Stack sizes (6 players)
        stack_sizes = np.array([p.stack / self.starting_stack for p in self.players])
        
        # Position info (one-hot)
        position_encoding = np.zeros(6)
        position_encoding[player.position.value] = 1
        
        # Pot odds
        pot_odds = np.array([self._get_pot_odds(player)])
        
        # Hand strength
        hand_strength = np.array([self._get_hand_strength(player)])
        
        observation = np.concatenate([
            hole_encoding,
            community_encoding, 
            betting_history,
            stack_sizes,
            position_encoding,
            pot_odds,
            hand_strength
        ])
        
        return observation.astype(np.float32)
    
    def _is_action_valid(self, action: int, player_idx: int) -> bool:
        player = self.players[player_idx]
        
        if player.folded or player.all_in:
            return False
            
        call_amount = self.current_bet - player.current_bet
        
        if action == Action.FOLD.value:
            return call_amount > 0
        elif action == Action.CHECK.value:
            return call_amount == 0
        elif action == Action.CALL.value:
            return call_amount > 0 and player.stack >= call_amount
        else:  # Betting actions
            bet_sizes = {
                Action.BET_25.value: int(self.pot * 0.25),
                Action.BET_50.value: int(self.pot * 0.5),
                Action.BET_75.value: int(self.pot * 0.75),
                Action.BET_100.value: self.pot,
                Action.ALL_IN.value: player.stack
            }
            bet_amount = bet_sizes.get(action, 0)
            return player.stack >= bet_amount + call_amount
    
    def _execute_action(self, action: int, player_idx: int) -> float:
        player = self.players[player_idx]
        reward = 0
        
        if action == Action.FOLD.value:
            player.folded = True
            reward = -0.1
            
        elif action == Action.CHECK.value:
            pass  # No additional bet
            
        elif action == Action.CALL.value:
            call_amount = self.current_bet - player.current_bet
            actual_call = min(call_amount, player.stack)
            player.current_bet += actual_call
            player.stack -= actual_call
            self.pot += actual_call
            
        else:  # Betting actions
            call_amount = self.current_bet - player.current_bet
            
            bet_sizes = {
                Action.BET_25.value: int(self.pot * 0.25),
                Action.BET_50.value: int(self.pot * 0.5),
                Action.BET_75.value: int(self.pot * 0.75),
                Action.BET_100.value: self.pot,
                Action.ALL_IN.value: player.stack
            }
            
            bet_amount = bet_sizes[action]
            total_amount = call_amount + bet_amount
            actual_amount = min(total_amount, player.stack)
            
            player.current_bet += actual_amount
            player.stack -= actual_amount
            self.pot += actual_amount
            self.current_bet = max(self.current_bet, player.current_bet)
            
            if player.stack == 0:
                player.all_in = True
        
        self.betting_round_actions.append(action)
        player.actions_this_round.append(action)
        
        return reward
    
    def _advance_game_phase(self):
        if self.game_phase == GamePhase.PREFLOP:
            self.community_cards.extend([self.deck.pop() for _ in range(3)])  # Flop
            self.game_phase = GamePhase.FLOP
        elif self.game_phase == GamePhase.FLOP:
            self.community_cards.append(self.deck.pop())  # Turn
            self.game_phase = GamePhase.TURN
        elif self.game_phase == GamePhase.TURN:
            self.community_cards.append(self.deck.pop())  # River
            self.game_phase = GamePhase.RIVER
    
    def _determine_winner(self) -> int:
        active_players = [i for i, p in enumerate(self.players) if not p.folded]
        
        if len(active_players) == 1:
            return active_players[0]
        
        best_rank = float('inf')
        winner = -1
        
        for player_idx in active_players:
            player = self.players[player_idx]
            try:
                hole_cards_treys = [card.to_treys() for card in player.hole_cards]
                community_treys = [card.to_treys() for card in self.community_cards]
                rank = self.evaluator.evaluate(community_treys, hole_cards_treys)
                
                if rank < best_rank:
                    best_rank = rank
                    winner = player_idx
            except:
                continue
                
        return winner
    
    def reset(self):
        # Reset all players
        for player in self.players:
            player.reset_for_hand()
            
        # Reset game state
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        self.game_phase = GamePhase.PREFLOP
        self.betting_round_actions = []
        
        # Create new deck and deal
        self._create_deck()
        self._deal_hole_cards()
        self._post_blinds()
        
        # Set first player to act (after big blind)
        self.current_player = (self.dealer_button + 3) % self.num_players
        
        return self._get_observation(0)  # Return observation for player 0
    
    def step(self, action: int):
        reward = 0
        done = False
        
        # Execute action if valid
        if self._is_action_valid(action, self.current_player):
            reward = self._execute_action(action, self.current_player)
        else:
            reward = -1  # Penalty for invalid action
            
        # Check if betting round is complete
        active_players = [i for i, p in enumerate(self.players) if not p.folded and not p.all_in]
        
        if len(active_players) <= 1:
            done = True
            winner = self._determine_winner()
            if winner == 0:  # Player 0 wins
                reward += self.pot / self.big_blind  # Reward proportional to pot
            else:
                reward -= self.current_bet / self.big_blind  # Penalty for losing
        
        # Advance to next player or next phase
        if not done:
            self.current_player = (self.current_player + 1) % self.num_players
            
            # Skip folded/all-in players
            while (self.players[self.current_player].folded or 
                   self.players[self.current_player].all_in):
                self.current_player = (self.current_player + 1) % self.num_players
            
            # Check if we've completed a betting round
            if self._betting_round_complete():
                if self.game_phase == GamePhase.RIVER:
                    done = True
                    winner = self._determine_winner()
                    if winner == 0:
                        reward += self.pot / self.big_blind
                    else:
                        reward -= self.current_bet / self.big_blind
                else:
                    self._advance_game_phase()
                    self.current_bet = 0
                    for player in self.players:
                        player.current_bet = 0
        
        next_state = self._get_observation(0)
        info = {
            'pot': self.pot,
            'phase': self.game_phase.name,
            'community_cards': [str(card) for card in self.community_cards]
        }
        
        return next_state, reward, done, info
    
    def _betting_round_complete(self) -> bool:
        active_players = [p for p in self.players if not p.folded and not p.all_in]
        
        if len(active_players) <= 1:
            return True
            
        # Check if all active players have matched the current bet
        for player in active_players:
            if player.current_bet != self.current_bet:
                return False
                
        return True
    
    def render(self, mode='human'):
        print(f"\n=== Game Phase: {self.game_phase.name} ===")
        print(f"Pot: ${self.pot}")
        print(f"Community Cards: {[str(card) for card in self.community_cards]}")
        
        for i, player in enumerate(self.players):
            status = ""
            if player.folded:
                status = "FOLDED"
            elif player.all_in:
                status = "ALL-IN"
            elif i == self.current_player:
                status = "TO ACT"
                
            print(f"{player.name}: ${player.stack} (Bet: ${player.current_bet}) {status}")
            if i == 0 and player.hole_cards:  # Show player 0's cards
                print(f"  Cards: {[str(card) for card in player.hole_cards]}")
    
    def get_valid_actions(self, player_idx: int) -> List[int]:
        return [action for action in range(8) if self._is_action_valid(action, player_idx)]