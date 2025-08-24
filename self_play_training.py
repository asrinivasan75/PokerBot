import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
import random
from collections import deque, defaultdict
import pickle
import os
from advanced_poker_env import AdvancedPokerEnv, Action
from neural_networks import create_poker_model, compile_poker_model
import matplotlib.pyplot as plt

class OpponentModel:
    """Model for tracking opponent tendencies and patterns"""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.action_frequency = defaultdict(int)
        self.situation_actions = defaultdict(list)  # situation -> list of actions
        self.aggression_level = 0.5  # 0 = passive, 1 = aggressive
        self.bluff_frequency = 0.1   # Estimated bluff frequency
        self.fold_frequency = 0.3    # Estimated fold frequency
        self.games_observed = 0
        
    def update(self, situation: str, action: int, hand_strength: float):
        """Update opponent model with new observation"""
        self.action_frequency[action] += 1
        self.situation_actions[situation].append(action)
        self.games_observed += 1
        
        # Update aggression level (betting/raising vs checking/calling)
        if action in [3, 4, 5, 6, 7]:  # Betting actions
            self.aggression_level = min(1.0, self.aggression_level + 0.01)
        elif action in [0, 1]:  # Fold/Check
            self.aggression_level = max(0.0, self.aggression_level - 0.01)
            
        # Update bluff frequency (betting with weak hands)
        if action in [3, 4, 5, 6] and hand_strength < 0.3:
            self.bluff_frequency = min(1.0, self.bluff_frequency + 0.05)
        elif action == 0 and hand_strength < 0.3:  # Fold with weak hand
            self.fold_frequency = min(1.0, self.fold_frequency + 0.02)
    
    def predict_action(self, situation: str) -> int:
        """Predict opponent's action based on observed patterns"""
        if situation in self.situation_actions:
            actions = self.situation_actions[situation]
            if actions:
                # Return most common action in this situation
                return max(set(actions), key=actions.count)
        
        # Fallback: return action based on general tendencies
        if self.aggression_level > 0.7:
            return random.choice([3, 4, 5, 6])  # Aggressive betting
        elif self.aggression_level < 0.3:
            return random.choice([0, 1, 2])     # Passive play
        else:
            return 2  # Call
    
    def get_features(self) -> np.ndarray:
        """Get opponent features for neural network"""
        total_actions = sum(self.action_frequency.values()) or 1
        action_probs = [self.action_frequency[i] / total_actions for i in range(8)]
        
        return np.array([
            self.aggression_level,
            self.bluff_frequency,
            self.fold_frequency,
            self.games_observed / 1000.0,  # Normalized games count
            *action_probs
        ])

class PopulationBasedTraining:
    """Population-based training for diverse opponent strategies"""
    
    def __init__(self, population_size: int = 20, env: AdvancedPokerEnv = None):
        self.population_size = population_size
        self.env = env or AdvancedPokerEnv()
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with diverse models"""
        model_types = ["transformer", "cnn", "lstm"]
        
        for i in range(self.population_size):
            model_type = model_types[i % len(model_types)]
            model = create_poker_model(model_type)
            model = compile_poker_model(model)
            
            # Add some noise to initial weights for diversity
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    weights = layer.get_weights()
                    if weights:
                        noisy_weights = [w + np.random.normal(0, 0.01, w.shape) for w in weights]
                        layer.set_weights(noisy_weights)
            
            self.population.append(model)
            self.fitness_scores.append(0.0)
    
    def evaluate_population(self, num_games: int = 100) -> List[float]:
        """Evaluate fitness of entire population"""
        fitness_scores = []
        
        for i, model in enumerate(self.population):
            print(f"Evaluating model {i+1}/{self.population_size}")
            fitness = self._evaluate_model(model, num_games)
            fitness_scores.append(fitness)
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def _evaluate_model(self, model: tf.keras.Model, num_games: int) -> float:
        """Evaluate a single model's fitness"""
        total_reward = 0
        wins = 0
        
        for _ in range(num_games):
            state = self.env.reset()
            done = False
            game_reward = 0
            
            while not done:
                if self.env.current_player == 0:  # Our model
                    try:
                        policy, value = model.predict(state.reshape(1, -1), verbose=0)
                        valid_actions = self.env.get_valid_actions(0)
                        
                        # Mask invalid actions
                        masked_policy = policy[0].copy()
                        for j in range(len(masked_policy)):
                            if j not in valid_actions:
                                masked_policy[j] = 0
                        
                        if masked_policy.sum() > 0:
                            masked_policy /= masked_policy.sum()
                            action = np.random.choice(len(masked_policy), p=masked_policy)
                        else:
                            action = random.choice(valid_actions)
                    except:
                        valid_actions = self.env.get_valid_actions(0)
                        action = random.choice(valid_actions) if valid_actions else 0
                else:  # Random opponent
                    valid_actions = self.env.get_valid_actions(self.env.current_player)
                    action = random.choice(valid_actions) if valid_actions else 0
                
                next_state, reward, done, info = self.env.step(action)
                
                if self.env.current_player == 0:
                    game_reward += reward
                
                state = next_state
            
            total_reward += game_reward
            if game_reward > 0:
                wins += 1
        
        # Fitness combines average reward and win rate
        avg_reward = total_reward / num_games
        win_rate = wins / num_games
        fitness = avg_reward + win_rate * 10  # Weight win rate heavily
        
        return fitness
    
    def evolve_population(self, mutation_rate: float = 0.1, elite_ratio: float = 0.2):
        """Evolve population using genetic algorithm principles"""
        self.generation += 1
        print(f"\n=== Generation {self.generation} ===")
        
        # Select elite models
        elite_count = int(self.population_size * elite_ratio)
        elite_indices = np.argsort(self.fitness_scores)[-elite_count:]
        
        new_population = []
        new_fitness_scores = []
        
        # Keep elite models
        for idx in elite_indices:
            new_population.append(self.population[idx])
            new_fitness_scores.append(self.fitness_scores[idx])
        
        # Generate offspring through crossover and mutation
        while len(new_population) < self.population_size:
            # Select parents based on fitness
            parent1_idx = self._tournament_selection()
            parent2_idx = self._tournament_selection()
            
            # Create offspring through crossover
            offspring = self._crossover(
                self.population[parent1_idx], 
                self.population[parent2_idx]
            )
            
            # Apply mutation
            if random.random() < mutation_rate:
                self._mutate(offspring)
            
            new_population.append(offspring)
            new_fitness_scores.append(0.0)  # Will be evaluated next generation
        
        self.population = new_population
        self.fitness_scores = new_fitness_scores
    
    def _tournament_selection(self, tournament_size: int = 3) -> int:
        """Select parent using tournament selection"""
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def _crossover(self, parent1: tf.keras.Model, parent2: tf.keras.Model) -> tf.keras.Model:
        """Create offspring through neural network crossover"""
        # Create new model with same architecture as parent1
        offspring = tf.keras.models.clone_model(parent1)
        offspring.build(input_shape=parent1.input_shape)
        
        # Crossover weights layer by layer
        for i, layer in enumerate(offspring.layers):
            if hasattr(layer, 'kernel') and i < len(parent2.layers):
                parent1_weights = parent1.layers[i].get_weights()
                parent2_weights = parent2.layers[i].get_weights()
                
                if parent1_weights and parent2_weights:
                    # Random crossover of weights
                    offspring_weights = []
                    for w1, w2 in zip(parent1_weights, parent2_weights):
                        mask = np.random.random(w1.shape) < 0.5
                        offspring_weight = np.where(mask, w1, w2)
                        offspring_weights.append(offspring_weight)
                    
                    layer.set_weights(offspring_weights)
        
        return offspring
    
    def _mutate(self, model: tf.keras.Model, mutation_strength: float = 0.01):
        """Apply mutation to model weights"""
        for layer in model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.get_weights()
                if weights:
                    mutated_weights = []
                    for w in weights:
                        mutation = np.random.normal(0, mutation_strength, w.shape)
                        mutated_weights.append(w + mutation)
                    layer.set_weights(mutated_weights)
    
    def get_best_model(self) -> tf.keras.Model:
        """Get the best model from current population"""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]
    
    def save_population(self, directory: str):
        """Save entire population to directory"""
        os.makedirs(directory, exist_ok=True)
        
        for i, model in enumerate(self.population):
            model.save(f"{directory}/model_{i}.h5")
        
        # Save fitness scores and metadata
        metadata = {
            'fitness_scores': self.fitness_scores,
            'generation': self.generation,
            'population_size': self.population_size
        }
        
        with open(f"{directory}/metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

class SelfPlayTrainer:
    """Advanced self-play trainer with opponent modeling"""
    
    def __init__(self, model: tf.keras.Model, env: AdvancedPokerEnv):
        self.model = model
        self.env = env
        self.opponent_models = {}  # player_id -> OpponentModel
        self.training_history = []
        self.experience_buffer = deque(maxlen=10000)
        
    def train_with_self_play(self, iterations: int = 1000, 
                           update_frequency: int = 100) -> Dict:
        """Train model using self-play with opponent modeling"""
        
        for iteration in range(iterations):
            if iteration % 100 == 0:
                print(f"Self-play iteration {iteration}/{iterations}")
            
            # Play a game and collect experience
            experience = self._play_self_play_game()
            self.experience_buffer.extend(experience)
            
            # Update model periodically
            if iteration % update_frequency == 0 and len(self.experience_buffer) > 1000:
                loss = self._update_model()
                self.training_history.append({
                    'iteration': iteration,
                    'loss': loss,
                    'experience_buffer_size': len(self.experience_buffer)
                })
        
        return {'history': self.training_history}
    
    def _play_self_play_game(self) -> List[Tuple]:
        """Play a game and return experience tuples"""
        experience = []
        states_actions = []  # Track for opponent modeling
        
        state = self.env.reset()
        done = False
        
        while not done:
            current_player = self.env.current_player
            
            # Get action based on current player
            if current_player == 0:  # Main model
                action = self._get_model_action(state)
            else:  # Opponent (could be another version of model or different strategy)
                action = self._get_opponent_action(state, current_player)
            
            # Track for opponent modeling
            hand_strength = self.env._get_hand_strength(self.env.players[current_player])
            situation = self._get_situation_key(state)
            states_actions.append((current_player, situation, action, hand_strength))
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience for main player (player 0)
            if current_player == 0:
                experience.append((state, action, reward, next_state, done))
            
            state = next_state
        
        # Update opponent models
        self._update_opponent_models(states_actions)
        
        return experience
    
    def _get_model_action(self, state: np.ndarray) -> int:
        """Get action from main model"""
        try:
            policy, value = self.model.predict(state.reshape(1, -1), verbose=0)
            valid_actions = self.env.get_valid_actions(self.env.current_player)
            
            # Add exploration noise
            exploration_rate = 0.1
            if random.random() < exploration_rate:
                return random.choice(valid_actions)
            
            # Mask invalid actions
            masked_policy = policy[0].copy()
            for i in range(len(masked_policy)):
                if i not in valid_actions:
                    masked_policy[i] = 0
            
            if masked_policy.sum() > 0:
                masked_policy /= masked_policy.sum()
                action = np.random.choice(len(masked_policy), p=masked_policy)
            else:
                action = random.choice(valid_actions)
                
            return action
        except:
            valid_actions = self.env.get_valid_actions(self.env.current_player)
            return random.choice(valid_actions) if valid_actions else 0
    
    def _get_opponent_action(self, state: np.ndarray, player_id: int) -> int:
        """Get action for opponent using opponent model or random"""
        valid_actions = self.env.get_valid_actions(player_id)
        
        if player_id in self.opponent_models:
            situation = self._get_situation_key(state)
            predicted_action = self.opponent_models[player_id].predict_action(situation)
            if predicted_action in valid_actions:
                return predicted_action
        
        # Fallback to random or model-based action
        return random.choice(valid_actions) if valid_actions else 0
    
    def _get_situation_key(self, state: np.ndarray) -> str:
        """Create situation key for opponent modeling"""
        # Simplify state to key features
        pot_odds = state[-2] if len(state) > 2 else 0
        hand_strength = state[-1] if len(state) > 1 else 0
        
        pot_bucket = int(pot_odds * 5)  # 0-5 buckets
        hand_bucket = int(hand_strength * 10)  # 0-10 buckets
        
        return f"pot_{pot_bucket}_hand_{hand_bucket}"
    
    def _update_opponent_models(self, states_actions: List[Tuple]):
        """Update opponent models with observations"""
        for player_id, situation, action, hand_strength in states_actions:
            if player_id != 0:  # Don't model ourselves
                if player_id not in self.opponent_models:
                    self.opponent_models[player_id] = OpponentModel(player_id)
                
                self.opponent_models[player_id].update(situation, action, hand_strength)
    
    def _update_model(self) -> float:
        """Update model using experience buffer"""
        if len(self.experience_buffer) < 32:
            return 0.0
        
        # Sample batch from experience buffer
        batch_size = min(32, len(self.experience_buffer))
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # Create target policies (one-hot for chosen actions)
        target_policies = np.zeros((batch_size, self.env.action_space.n))
        for i, action in enumerate(actions):
            target_policies[i, action] = 1.0
        
        # Calculate target values using TD learning
        target_values = rewards.copy()
        for i in range(batch_size):
            if not dones[i]:
                try:
                    _, next_value = self.model.predict(next_states[i].reshape(1, -1), verbose=0)
                    target_values[i] += 0.99 * next_value[0, 0]  # Gamma = 0.99
                except:
                    pass
        
        # Train model
        history = self.model.fit(
            states,
            {'policy': target_policies, 'value': target_values.reshape(-1, 1)},
            epochs=1,
            verbose=0
        )
        
        return history.history['loss'][0]
    
    def plot_training_progress(self):
        """Plot training progress"""
        if not self.training_history:
            print("No training history available")
            return
        
        iterations = [h['iteration'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

def run_population_training(generations: int = 10, population_size: int = 20):
    """Run complete population-based training"""
    env = AdvancedPokerEnv()
    pbt = PopulationBasedTraining(population_size, env)
    
    for generation in range(generations):
        print(f"\n=== Generation {generation + 1}/{generations} ===")
        
        # Evaluate population
        fitness_scores = pbt.evaluate_population(num_games=50)
        
        print(f"Best fitness: {max(fitness_scores):.3f}")
        print(f"Average fitness: {np.mean(fitness_scores):.3f}")
        
        # Evolve if not last generation
        if generation < generations - 1:
            pbt.evolve_population()
    
    # Save best model
    best_model = pbt.get_best_model()
    best_model.save("best_poker_model.h5")
    
    # Save entire population
    pbt.save_population("poker_population")
    
    print("Population training complete!")
    return pbt

if __name__ == "__main__":
    # Example usage
    print("Starting population-based training...")
    pbt = run_population_training(generations=5, population_size=10)
    
    print("\nStarting self-play training with best model...")
    env = AdvancedPokerEnv()
    best_model = pbt.get_best_model()
    
    trainer = SelfPlayTrainer(best_model, env)
    history = trainer.train_with_self_play(iterations=500)
    
    # Save final model
    best_model.save("final_poker_model.h5")
    
    print("Training complete!")