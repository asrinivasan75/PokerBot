import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Conv1D, LSTM, Attention, 
                                   MultiHeadAttention, LayerNormalization,
                                   Dropout, Input, Concatenate, Flatten,
                                   BatchNormalization, Embedding)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple, List, Dict
from advanced_poker_env import AdvancedPokerEnv

class TransformerBlock(tf.keras.layers.Layer):
    """Transformer block for sequence modeling"""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PokerTransformer(Model):
    """Transformer-based poker neural network"""
    
    def __init__(self, num_actions: int = 8, embed_dim: int = 128, 
                 num_heads: int = 8, ff_dim: int = 256, num_blocks: int = 4):
        super(PokerTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Input processing layers
        self.card_embedding = Dense(embed_dim, activation='relu', name='card_embedding')
        self.betting_embedding = Dense(embed_dim, activation='relu', name='betting_embedding')
        self.position_embedding = Dense(embed_dim, activation='relu', name='position_embedding')
        
        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_blocks)
        ]
        
        # Output layers
        self.dropout = Dropout(0.1)
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))
        self.value_head = Dense(1, activation='tanh', name='value')  # State value
        self.policy_head = Dense(num_actions, activation='softmax', name='policy')  # Action probabilities
        
    def call(self, inputs, training=None):
        # Parse inputs (observation vector of size 428)
        cards = inputs[:, :364]  # Card encodings (hole + community)
        betting = inputs[:, 364:414]  # Betting history
        position = inputs[:, 414:420]  # Position encoding
        pot_odds = inputs[:, 420:421]  # Pot odds
        hand_strength = inputs[:, 421:422]  # Hand strength
        
        # Embed different input types
        card_embed = self.card_embedding(cards)
        betting_embed = self.betting_embedding(betting)
        position_embed = self.position_embedding(position)
        
        # Combine embeddings
        combined = tf.stack([card_embed, betting_embed, position_embed], axis=1)
        
        # Apply transformer blocks
        x = combined
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)
        
        # Flatten and process
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        
        # Add pot odds and hand strength
        additional_features = tf.concat([pot_odds, hand_strength], axis=1)
        x = tf.concat([x, additional_features], axis=1)
        
        # Final dense layers
        x = self.dense1(x)
        x = self.dense2(x)
        
        # Output heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        return policy, value

class ConvolutionalPokerNet(Model):
    """CNN-based poker network for pattern recognition"""
    
    def __init__(self, num_actions: int = 8):
        super(ConvolutionalPokerNet, self).__init__()
        
        # Convolutional layers for card pattern recognition
        self.conv1 = Conv1D(64, 3, activation='relu', padding='same')
        self.conv2 = Conv1D(128, 3, activation='relu', padding='same')
        self.conv3 = Conv1D(256, 3, activation='relu', padding='same')
        
        # Batch normalization
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        
        # Dense layers
        self.flatten = Flatten()
        self.dense1 = Dense(512, activation='relu', kernel_regularizer=l2(0.001))
        self.dense2 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.dropout = Dropout(0.3)
        
        # Output heads
        self.value_head = Dense(1, activation='tanh', name='value')
        self.policy_head = Dense(num_actions, activation='softmax', name='policy')
        
    def call(self, inputs, training=None):
        # Reshape for 1D convolution
        x = tf.expand_dims(inputs, axis=2)
        
        # Convolutional layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        
        # Flatten and dense layers
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        # Output heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        return policy, value

class LSTMPokerNet(Model):
    """LSTM-based network for sequential decision making"""
    
    def __init__(self, num_actions: int = 8, lstm_units: int = 128):
        super(LSTMPokerNet, self).__init__()
        
        # LSTM layers for sequence modeling
        self.lstm1 = LSTM(lstm_units, return_sequences=True, dropout=0.2)
        self.lstm2 = LSTM(lstm_units, dropout=0.2)
        
        # Dense layers
        self.dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.001))
        self.dense2 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))
        self.dropout = Dropout(0.3)
        
        # Output heads
        self.value_head = Dense(1, activation='tanh', name='value')
        self.policy_head = Dense(num_actions, activation='softmax', name='policy')
        
    def call(self, inputs, training=None):
        # Reshape for LSTM (sequence of features)
        sequence_length = 10  # Use last 10 "time steps" of features
        feature_dim = inputs.shape[-1] // sequence_length
        
        # Pad if necessary
        if inputs.shape[-1] % sequence_length != 0:
            padding_size = sequence_length - (inputs.shape[-1] % sequence_length)
            padding = tf.zeros((tf.shape(inputs)[0], padding_size))
            inputs = tf.concat([inputs, padding], axis=1)
            feature_dim = inputs.shape[-1] // sequence_length
        
        x = tf.reshape(inputs, (-1, sequence_length, feature_dim))
        
        # LSTM layers
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Dense layers
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        
        # Output heads
        value = self.value_head(x)
        policy = self.policy_head(x)
        
        return policy, value

class EnsemblePokerNet(Model):
    """Ensemble of different network architectures"""
    
    def __init__(self, num_actions: int = 8):
        super(EnsemblePokerNet, self).__init__()
        
        # Different network architectures
        self.transformer = PokerTransformer(num_actions)
        self.cnn = ConvolutionalPokerNet(num_actions)
        self.lstm = LSTMPokerNet(num_actions)
        
        # Ensemble combination layers
        self.policy_combiner = Dense(num_actions, activation='softmax', name='ensemble_policy')
        self.value_combiner = Dense(1, activation='tanh', name='ensemble_value')
        
    def call(self, inputs, training=None):
        # Get predictions from each network
        transformer_policy, transformer_value = self.transformer(inputs, training=training)
        cnn_policy, cnn_value = self.cnn(inputs, training=training)
        lstm_policy, lstm_value = self.lstm(inputs, training=training)
        
        # Combine policies (weighted average)
        combined_policy = (transformer_policy + cnn_policy + lstm_policy) / 3.0
        combined_value = (transformer_value + cnn_value + lstm_value) / 3.0
        
        return combined_policy, combined_value

class OpponentModelingNet(Model):
    """Network for modeling opponent behavior"""
    
    def __init__(self, num_players: int = 6, num_actions: int = 8):
        super(OpponentModelingNet, self).__init__()
        
        self.num_players = num_players
        
        # Opponent behavior embedding
        self.opponent_embedding = Dense(64, activation='relu')
        
        # Player-specific networks
        self.player_networks = [
            Sequential([
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(num_actions, activation='softmax')
            ]) for _ in range(num_players)
        ]
        
        # Aggregation layer
        self.aggregation = Dense(num_actions, activation='softmax')
        
    def call(self, inputs, current_player: int = 0, training=None):
        # Extract opponent features
        opponent_features = self.opponent_embedding(inputs)
        
        # Get predictions for each player
        player_predictions = []
        for i, player_net in enumerate(self.player_networks):
            pred = player_net(opponent_features, training=training)
            player_predictions.append(pred)
        
        # Return prediction for current player
        return player_predictions[current_player]

def create_poker_model(model_type: str = "transformer", num_actions: int = 8, 
                      input_shape: Tuple[int] = (428,)) -> Model:
    """Factory function to create different poker models"""
    
    if model_type == "transformer":
        return PokerTransformer(num_actions)
    elif model_type == "cnn":
        return ConvolutionalPokerNet(num_actions)
    elif model_type == "lstm":
        return LSTMPokerNet(num_actions)
    elif model_type == "ensemble":
        return EnsemblePokerNet(num_actions)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def compile_poker_model(model: Model, learning_rate: float = 0.001) -> Model:
    """Compile poker model with appropriate loss functions"""
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        loss_weights={'policy': 1.0, 'value': 0.5},
        metrics={
            'policy': 'categorical_accuracy',
            'value': 'mae'
        }
    )
    
    return model

class PokerModelTrainer:
    """Trainer for poker neural networks with self-play"""
    
    def __init__(self, model: Model, env: AdvancedPokerEnv):
        self.model = model
        self.env = env
        self.training_data = []
        
    def generate_training_data(self, num_games: int = 1000) -> List[Tuple]:
        """Generate training data through self-play"""
        training_data = []
        
        for game in range(num_games):
            if game % 100 == 0:
                print(f"Generating game {game}/{num_games}")
                
            states, actions, rewards = self._play_game()
            
            # Convert to training examples
            for i, (state, action, reward) in enumerate(zip(states, actions, rewards)):
                # Create target policy (one-hot for chosen action)
                target_policy = np.zeros(self.env.action_space.n)
                target_policy[action] = 1.0
                
                # Use discounted future reward as value target
                discounted_reward = sum(r * (0.99 ** j) for j, r in enumerate(rewards[i:]))
                
                training_data.append((state, target_policy, discounted_reward))
        
        return training_data
    
    def _play_game(self) -> Tuple[List, List, List]:
        """Play a single game and collect experience"""
        states, actions, rewards = [], [], []
        
        state = self.env.reset()
        done = False
        
        while not done:
            # Get action from model
            if hasattr(self.model, 'predict'):
                policy, value = self.model.predict(state.reshape(1, -1), verbose=0)
                valid_actions = self.env.get_valid_actions(self.env.current_player)
                
                # Mask invalid actions
                masked_policy = policy[0].copy()
                for i in range(len(masked_policy)):
                    if i not in valid_actions:
                        masked_policy[i] = 0
                
                # Renormalize
                if masked_policy.sum() > 0:
                    masked_policy /= masked_policy.sum()
                    action = np.random.choice(len(masked_policy), p=masked_policy)
                else:
                    action = np.random.choice(valid_actions)
            else:
                valid_actions = self.env.get_valid_actions(self.env.current_player)
                action = np.random.choice(valid_actions) if valid_actions else 0
            
            states.append(state.copy())
            actions.append(action)
            
            next_state, reward, done, info = self.env.step(action)
            rewards.append(reward)
            
            state = next_state
        
        return states, actions, rewards
    
    def train(self, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train the model using self-play data"""
        
        # Generate training data
        print("Generating training data...")
        training_data = self.generate_training_data()
        
        # Prepare training arrays
        states = np.array([d[0] for d in training_data])
        target_policies = np.array([d[1] for d in training_data])
        target_values = np.array([d[2] for d in training_data])
        
        # Train model
        print("Training model...")
        history = self.model.fit(
            states,
            {'policy': target_policies, 'value': target_values},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history.history

if __name__ == "__main__":
    # Example usage
    env = AdvancedPokerEnv()
    
    # Create and compile different models
    transformer_model = create_poker_model("transformer")
    transformer_model = compile_poker_model(transformer_model)
    
    # Train model
    trainer = PokerModelTrainer(transformer_model, env)
    history = trainer.train(epochs=10)
    
    print("Training complete!")
    transformer_model.save("poker_transformer.h5")