import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import random
from advanced_poker_env import AdvancedPokerEnv, Action
import pickle
import os

class InfoSet:
    """Information set for CFR algorithm"""
    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)
        self.strategy = np.ones(num_actions) / num_actions
        
    def get_strategy(self, realization_weight: float = 1.0) -> np.ndarray:
        """Get current strategy using regret matching"""
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        
        if normalizing_sum > 0:
            self.strategy = positive_regrets / normalizing_sum
        else:
            self.strategy = np.ones(self.num_actions) / self.num_actions
            
        self.strategy_sum += realization_weight * self.strategy
        return self.strategy
    
    def get_average_strategy(self) -> np.ndarray:
        """Get average strategy over all iterations"""
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        else:
            return np.ones(self.num_actions) / self.num_actions

class CFRTrainer:
    """Counterfactual Regret Minimization trainer for poker"""
    
    def __init__(self, env: AdvancedPokerEnv):
        self.env = env
        self.info_sets: Dict[str, InfoSet] = {}
        self.num_actions = env.action_space.n
        
    def get_info_set_key(self, observation: np.ndarray, valid_actions: List[int]) -> str:
        """Create a key for the information set based on observation"""
        # Simplify observation to create manageable info sets
        # Use hand strength, pot odds, and betting pattern
        hand_strength = observation[-1]  # Last feature is hand strength
        pot_odds = observation[-2]       # Second to last is pot odds
        
        # Discretize continuous values
        hand_bucket = int(hand_strength * 10)  # 0-10 buckets
        pot_bucket = int(pot_odds * 5)         # 0-5 buckets
        
        # Include betting history pattern (simplified)
        betting_pattern = str(int(np.sum(observation[364:414]) % 100))  # Betting history sum mod 100
        
        # Include valid actions
        valid_actions_str = ''.join(map(str, sorted(valid_actions)))
        
        return f"h{hand_bucket}_p{pot_bucket}_b{betting_pattern}_a{valid_actions_str}"
    
    def get_info_set(self, key: str) -> InfoSet:
        """Get or create info set for given key"""
        if key not in self.info_sets:
            self.info_sets[key] = InfoSet(self.num_actions)
        return self.info_sets[key]
    
    def cfr(self, player: int = 0, iterations: int = 1000) -> Dict[str, np.ndarray]:
        """Run CFR algorithm for specified iterations"""
        
        for iteration in range(iterations):
            if iteration % 100 == 0:
                print(f"CFR Iteration {iteration}/{iterations}")
                
            # Reset environment for new hand
            self.env.reset()
            
            # Run CFR for this hand
            self._cfr_recursive(player, 1.0, 1.0)
            
        # Return average strategies
        strategies = {}
        for key, info_set in self.info_sets.items():
            strategies[key] = info_set.get_average_strategy()
            
        return strategies
    
    def _cfr_recursive(self, player: int, p0: float, p1: float) -> float:
        """Recursive CFR implementation"""
        
        # Get current observation and valid actions
        observation = self.env._get_observation(player)
        valid_actions = self.env.get_valid_actions(player)
        
        # Check if game is terminal
        if len(valid_actions) == 0 or self._is_terminal():
            return self._get_utility(player)
        
        # Get information set
        info_set_key = self.get_info_set_key(observation, valid_actions)
        info_set = self.get_info_set(info_set_key)
        
        # Get current strategy
        if player == 0:
            strategy = info_set.get_strategy(p0)
        else:
            strategy = info_set.get_strategy(p1)
        
        # Initialize utilities and counter-factual values
        util = np.zeros(self.num_actions)
        node_util = 0.0
        
        # For each valid action
        for i, action in enumerate(valid_actions):
            if action >= self.num_actions:
                continue
                
            # Save current state
            old_state = self._save_state()
            
            # Take action
            next_obs, reward, done, info = self.env.step(action)
            
            if player == 0:
                util[action] = -self._cfr_recursive(1 - player, p0 * strategy[action], p1)
            else:
                util[action] = -self._cfr_recursive(1 - player, p0, p1 * strategy[action])
                
            node_util += strategy[action] * util[action]
            
            # Restore state
            self._restore_state(old_state)
        
        # Update regrets for current player
        if player == 0:
            for action in valid_actions:
                if action < self.num_actions:
                    regret = util[action] - node_util
                    info_set.regret_sum[action] += p1 * regret
        else:
            for action in valid_actions:
                if action < self.num_actions:
                    regret = util[action] - node_util
                    info_set.regret_sum[action] += p0 * regret
        
        return node_util
    
    def _is_terminal(self) -> bool:
        """Check if current state is terminal"""
        active_players = sum(1 for p in self.env.players if not p.folded)
        return active_players <= 1 or self.env.game_phase.value > 3
    
    def _get_utility(self, player: int) -> float:
        """Get utility for terminal state"""
        if self.env.players[player].folded:
            return -self.env.players[player].current_bet
        
        winner = self.env._determine_winner()
        if winner == player:
            return self.env.pot - self.env.players[player].current_bet
        else:
            return -self.env.players[player].current_bet
    
    def _save_state(self) -> Dict:
        """Save current environment state"""
        return {
            'current_player': self.env.current_player,
            'pot': self.env.pot,
            'current_bet': self.env.current_bet,
            'game_phase': self.env.game_phase,
            'community_cards': self.env.community_cards.copy(),
            'player_states': [(p.stack, p.current_bet, p.folded, p.all_in) 
                             for p in self.env.players]
        }
    
    def _restore_state(self, state: Dict):
        """Restore environment state"""
        self.env.current_player = state['current_player']
        self.env.pot = state['pot']
        self.env.current_bet = state['current_bet']
        self.env.game_phase = state['game_phase']
        self.env.community_cards = state['community_cards']
        
        for i, (stack, bet, folded, all_in) in enumerate(state['player_states']):
            self.env.players[i].stack = stack
            self.env.players[i].current_bet = bet
            self.env.players[i].folded = folded
            self.env.players[i].all_in = all_in
    
    def save_strategy(self, filename: str):
        """Save trained strategy to file"""
        strategies = {}
        for key, info_set in self.info_sets.items():
            strategies[key] = info_set.get_average_strategy()
        
        with open(filename, 'wb') as f:
            pickle.dump(strategies, f)
        print(f"Strategy saved to {filename}")
    
    def load_strategy(self, filename: str) -> Dict[str, np.ndarray]:
        """Load strategy from file"""
        with open(filename, 'rb') as f:
            strategies = pickle.load(f)
        print(f"Strategy loaded from {filename}")
        return strategies

class CFRAgent:
    """Agent that uses CFR-trained strategy"""
    
    def __init__(self, strategy_file: str = None):
        self.strategies = {}
        if strategy_file and os.path.exists(strategy_file):
            with open(strategy_file, 'rb') as f:
                self.strategies = pickle.load(f)
    
    def get_action(self, env: AdvancedPokerEnv, player: int = 0) -> int:
        """Get action using CFR strategy"""
        observation = env._get_observation(player)
        valid_actions = env.get_valid_actions(player)
        
        if not valid_actions:
            return 0  # Fold if no valid actions
        
        # Create info set key
        trainer = CFRTrainer(env)  # Temporary for key generation
        info_set_key = trainer.get_info_set_key(observation, valid_actions)
        
        if info_set_key in self.strategies:
            strategy = self.strategies[info_set_key]
            
            # Sample action based on strategy
            action_probs = []
            for action in range(len(strategy)):
                if action in valid_actions:
                    action_probs.append(strategy[action])
                else:
                    action_probs.append(0.0)
            
            # Normalize probabilities for valid actions only
            valid_probs = [action_probs[a] for a in valid_actions]
            if sum(valid_probs) > 0:
                valid_probs = np.array(valid_probs)
                valid_probs /= valid_probs.sum()
                chosen_idx = np.random.choice(len(valid_actions), p=valid_probs)
                return valid_actions[chosen_idx]
        
        # Fallback: random valid action
        return random.choice(valid_actions)

def train_cfr_agent(iterations: int = 10000, save_path: str = "cfr_strategy.pkl"):
    """Train a CFR agent and save the strategy"""
    env = AdvancedPokerEnv()
    trainer = CFRTrainer(env)
    
    print(f"Training CFR agent for {iterations} iterations...")
    strategies = trainer.cfr(iterations=iterations)
    
    trainer.save_strategy(save_path)
    print(f"Training complete! Strategy saved to {save_path}")
    
    return trainer, strategies

if __name__ == "__main__":
    # Example usage
    trainer, strategies = train_cfr_agent(iterations=1000)
    
    # Test the trained agent
    env = AdvancedPokerEnv()
    agent = CFRAgent("cfr_strategy.pkl")
    
    # Play a few hands
    for hand in range(5):
        state = env.reset()
        done = False
        print(f"\n=== Hand {hand + 1} ===")
        
        while not done:
            if env.current_player == 0:  # CFR agent's turn
                action = agent.get_action(env, 0)
                print(f"CFR Agent action: {Action(action).name}")
            else:  # Random opponent
                valid_actions = env.get_valid_actions(env.current_player)
                action = random.choice(valid_actions) if valid_actions else 0
                print(f"Opponent action: {Action(action).name}")
            
            state, reward, done, info = env.step(action)
            env.render()
            
            if done:
                print(f"Hand complete! Reward: {reward}")
                break