#!/usr/bin/env python3
"""
Simple Reinforcement Learning Poker Bot
Implements basic call/raise/fold actions with reward shaping for strategy convergence
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict
import pickle
import matplotlib.pyplot as plt

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

class SimplePokerEnv:
    """Simple poker environment for RL training"""
    
    def __init__(self):
        self.hand_count = 0
        self.reset()
        
    def reset(self) -> np.ndarray:
        """Reset environment for new hand"""
        # Simple state: [hand_strength, pot_odds, position, opponent_aggression]
        self.hand_strength = random.uniform(0, 1)  # 0=worst, 1=best
        self.pot = 20  # Starting pot (blinds)
        self.player_bet = 10  # Small blind
        self.opponent_bet = 20  # Big blind  
        self.position = random.choice([0, 1])  # 0=early, 1=late
        self.opponent_aggression = random.uniform(0, 1)  # Historical aggression
        self.game_over = False
        self.hand_count += 1
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        pot_odds = self.opponent_bet / (self.pot + self.opponent_bet) if (self.pot + self.opponent_bet) > 0 else 0
        return np.array([
            self.hand_strength,
            pot_odds,
            self.position,
            self.opponent_aggression
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return next state, reward, done, info"""
        reward = 0
        info = {'action_taken': Action(action).name}
        
        if action == Action.FOLD.value:
            # REWARD SHAPING: Penalize folding good hands, reward folding bad hands
            if self.hand_strength > 0.7:
                reward = -10  # Bad fold of strong hand
            elif self.hand_strength < 0.3:
                reward = 5   # Good fold of weak hand
            else:
                reward = -2  # Neutral fold
            self.game_over = True
            info['result'] = 'fold'
            
        elif action == Action.CALL.value:
            call_amount = self.opponent_bet - self.player_bet
            self.player_bet = self.opponent_bet
            
            # Opponent decision (simplified)
            if random.random() < self.opponent_aggression:
                # Opponent raises
                raise_amount = random.randint(10, 30)
                self.opponent_bet += raise_amount
                self.pot += call_amount  # Add our call to pot
                reward = -1  # Small penalty for being raised
            else:
                # Showdown
                self.pot += call_amount
                if self.hand_strength > random.uniform(0, 1):  # Win
                    reward = self.pot * 0.5  # REWARD SHAPING: Scale with pot size
                    info['result'] = 'win'
                else:  # Lose
                    reward = -self.player_bet
                    info['result'] = 'lose'
                self.game_over = True
                
        elif action == Action.RAISE.value:
            # REWARD SHAPING: Encourage raising with strong hands
            if self.hand_strength > 0.6:
                base_reward = 5  # Bonus for raising with strong hand
            else:
                base_reward = -3  # Penalty for raising with weak hand
                
            raise_amount = random.randint(15, 40)
            self.player_bet += raise_amount
            self.pot += raise_amount
            
            # Opponent response
            if random.random() < (self.opponent_aggression * self.hand_strength):
                # Opponent calls
                if self.hand_strength > random.uniform(0, 1):
                    reward = base_reward + self.pot * 0.3
                    info['result'] = 'win_after_raise'
                else:
                    reward = base_reward - self.player_bet
                    info['result'] = 'lose_after_raise'
                self.game_over = True
            else:
                # Opponent folds - we win pot
                reward = base_reward + self.pot * 0.8  # Big reward for successful bluff/value bet
                info['result'] = 'opponent_fold'
                self.game_over = True
        
        return self._get_state(), reward, self.game_over, info

class SimpleRLAgent:
    """Simple Q-learning based poker agent"""
    
    def __init__(self, state_size=4, action_size=3, learning_rate=0.01, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Simple neural network using numpy
        self.W1 = np.random.randn(state_size, 64) * 0.1
        self.b1 = np.zeros((1, 64))
        self.W2 = np.random.randn(64, 32) * 0.1
        self.b2 = np.zeros((1, 32))
        self.W3 = np.random.randn(32, action_size) * 0.1
        self.b3 = np.zeros((1, action_size))
        
        # Training history
        self.rewards_history = []
        self.epsilon_history = []
        
    def _forward(self, state):
        """Forward pass through network"""
        z1 = np.dot(state.reshape(1, -1), self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        z3 = np.dot(a2, self.W3) + self.b3
        return z3, a2, a1
        
    def predict(self, state):
        """Predict Q-values for given state"""
        q_values, _, _ = self._forward(state)
        return q_values[0]
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.predict(state)
        return np.argmax(q_values)
    
    def train(self, state, action, reward, next_state, done, discount_factor=0.95):
        """Train the agent using Q-learning"""
        # Forward pass
        q_values, a2, a1 = self._forward(state)
        
        # Calculate target
        if done:
            target = reward
        else:
            next_q_values = self.predict(next_state)
            target = reward + discount_factor * np.max(next_q_values)
        
        # Calculate loss and gradients
        target_q = q_values.copy()
        target_q[0, action] = target
        
        # Backward pass (simplified)
        dW3 = np.dot(a2.T, (q_values - target_q))
        db3 = np.sum(q_values - target_q, axis=0, keepdims=True)
        
        # Update weights
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=10000, save_path='poker_agent.pkl'):
    """Train the RL agent over multiple episodes"""
    env = SimplePokerEnv()
    agent = SimpleRLAgent()
    
    episode_rewards = []
    recent_rewards = []
    
    print("Training RL Poker Agent...")
    print("Using reward shaping to improve strategy convergence")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.game_over:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        episode_rewards.append(total_reward)
        recent_rewards.append(total_reward)
        
        # Keep only recent 1000 episodes for averaging
        if len(recent_rewards) > 1000:
            recent_rewards.pop(0)
            
        if episode % 1000 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Store metrics
            agent.rewards_history.append(avg_reward)
            agent.epsilon_history.append(agent.epsilon)
    
    # Save trained agent
    agent_data = {
        'W1': agent.W1, 'b1': agent.b1,
        'W2': agent.W2, 'b2': agent.b2, 
        'W3': agent.W3, 'b3': agent.b3,
        'rewards_history': agent.rewards_history,
        'epsilon_history': agent.epsilon_history,
        'final_epsilon': agent.epsilon
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(agent_data, f)
    
    return agent, episode_rewards

def evaluate_agent(agent, episodes=1000):
    """Evaluate trained agent performance"""
    env = SimplePokerEnv()
    
    # Test trained agent
    agent.epsilon = 0  # No exploration during evaluation
    trained_rewards = []
    
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.game_over:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        trained_rewards.append(total_reward)
    
    # Test random agent for comparison
    random_rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while not env.game_over:
            action = random.randint(0, 2)  # Random action
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            
        random_rewards.append(total_reward)
    
    # Calculate improvement
    trained_avg = np.mean(trained_rewards)
    random_avg = np.mean(random_rewards)
    improvement = ((trained_avg - random_avg) / abs(random_avg)) * 100
    
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Trained Agent Avg Reward: {trained_avg:.2f}")
    print(f"Random Agent Avg Reward: {random_avg:.2f}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Training episodes: {episodes}")
    
    return improvement, trained_avg, random_avg

def plot_training_progress(agent):
    """Plot training progress"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(agent.rewards_history)
    plt.title('Average Reward During Training')
    plt.xlabel('Training Checkpoint (x1000 episodes)')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.epsilon_history)
    plt.title('Exploration Rate (Epsilon) Decay')
    plt.xlabel('Training Checkpoint (x1000 episodes)')
    plt.ylabel('Epsilon')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("Training progress saved as 'training_progress.png'")

if __name__ == "__main__":
    # Train agent
    agent, rewards = train_agent(episodes=10000)
    
    # Evaluate performance
    improvement, trained_avg, random_avg = evaluate_agent(agent)
    
    # Plot results
    plot_training_progress(agent)
    
    print(f"\n🎯 SUCCESS: Achieved {improvement:.1f}% improvement over random play")
    print("💾 Agent saved as 'poker_agent.pkl'")
    print("📊 Training plots saved as 'training_progress.png'")