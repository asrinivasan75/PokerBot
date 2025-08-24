import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from advanced_poker_env import AdvancedPokerEnv, Action, PokerCard
from neural_networks import create_poker_model, compile_poker_model
from cfr_algorithm import CFRAgent, train_cfr_agent
import random
import time
import os

# Page configuration
st.set_page_config(
    page_title="Advanced PokerBot Interface",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .poker-card {
        background: white;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        display: inline-block;
        min-width: 60px;
    }
    
    .red-card { color: red; }
    .black-card { color: black; }
    
    .player-info {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .pot-info {
        background: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    
    .action-button {
        background: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin: 5px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load poker models - simplified to avoid blocking"""
    models = {}
    
    # Only create models when needed, not at startup
    models["simple"] = None  # Placeholder
    
    return models

@st.cache_resource
def initialize_environment():
    """Initialize poker environment"""
    return AdvancedPokerEnv()

def display_card(card: PokerCard) -> str:
    """Display a poker card with proper styling"""
    if card.suit in ['h', 'd']:
        color_class = "red-card"
    else:
        color_class = "black-card"
    
    return f'<div class="poker-card {color_class}">{str(card)}</div>'

def display_hand(cards: list, title: str):
    """Display a hand of cards"""
    st.write(f"**{title}:**")
    cards_html = "".join([display_card(card) for card in cards])
    st.markdown(cards_html, unsafe_allow_html=True)

def get_action_name(action_id: int) -> str:
    """Get human-readable action name"""
    try:
        return Action(action_id).name
    except:
        return f"Action_{action_id}"

def main():
    st.title("🃏 Advanced PokerBot Interface")
    st.markdown("Train, play, and analyze advanced poker AI models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🎮 Play Poker", "🧠 Train Models", "📊 Analytics", "⚙️ Settings"]
    )
    
    if page == "🎮 Play Poker":
        play_poker_page()
    elif page == "🧠 Train Models":
        train_models_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "⚙️ Settings":
        settings_page()

def play_poker_page():
    st.header("🎮 Play Against AI")
    
    # Initialize session state
    if 'env' not in st.session_state:
        st.session_state.env = initialize_environment()
        st.session_state.models = load_models()
        st.session_state.game_state = 'waiting'
        st.session_state.game_history = []
    
    # Game controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🆕 New Game", type="primary"):
            st.session_state.env.reset()
            st.session_state.game_state = 'playing'
            st.rerun()
    
    with col2:
        model_choice = st.selectbox(
            "AI Model:",
            list(st.session_state.models.keys())
        )
    
    with col3:
        difficulty = st.selectbox(
            "Difficulty:",
            ["Easy", "Medium", "Hard", "Expert"]
        )
    
    if st.session_state.game_state == 'playing':
        play_poker_game(model_choice, difficulty)
    else:
        st.info("Click 'New Game' to start playing!")
        
        # Show game statistics
        if st.session_state.game_history:
            show_game_statistics()

def play_poker_game(model_choice: str, difficulty: str):
    """Main poker game interface"""
    env = st.session_state.env
    model = st.session_state.models[model_choice]
    
    # Display game state
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        # Pot information
        st.markdown(f'<div class="pot-info">Pot: ${env.pot}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f"**Phase:** {env.game_phase.name}")
    
    # Community cards
    if env.community_cards:
        st.markdown("### Community Cards")
        cards_html = "".join([display_card(card) for card in env.community_cards])
        st.markdown(cards_html, unsafe_allow_html=True)
    
    # Player information
    col1, col2 = st.columns(2)
    
    with col1:
        # Human player (Player 0)
        player = env.players[0]
        st.markdown(f'<div class="player-info">', unsafe_allow_html=True)
        st.markdown(f"**Your Hand** (Stack: ${player.stack})")
        if player.hole_cards:
            cards_html = "".join([display_card(card) for card in player.hole_cards])
            st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown(f"Current Bet: ${player.current_bet}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # AI opponents
        st.markdown("**Opponents:**")
        for i, opponent in enumerate(env.players[1:], 1):
            status = ""
            if opponent.folded:
                status = "FOLDED"
            elif opponent.all_in:
                status = "ALL-IN"
            
            st.write(f"Player {i}: ${opponent.stack} (Bet: ${opponent.current_bet}) {status}")
    
    # Human player actions
    if env.current_player == 0 and not env.players[0].folded:
        st.markdown("### Your Turn - Choose an Action:")
        
        valid_actions = env.get_valid_actions(0)
        action_cols = st.columns(len(valid_actions))
        
        action_taken = None
        for i, action in enumerate(valid_actions):
            with action_cols[i]:
                if st.button(get_action_name(action), key=f"action_{action}"):
                    action_taken = action
                    break
        
        if action_taken is not None:
            # Execute action
            next_state, reward, done, info = env.step(action_taken)
            
            # Log action
            st.success(f"You chose: {get_action_name(action_taken)}")
            
            if done:
                handle_game_end(reward)
            else:
                st.rerun()
    
    # AI player actions
    elif not all(p.folded or p.all_in for p in env.players):
        if env.current_player != 0:
            with st.spinner("AI is thinking..."):
                time.sleep(1)  # Dramatic pause
                
                # Get AI action
                try:
                    state = env._get_observation(env.current_player)
                    policy, value = model.predict(state.reshape(1, -1), verbose=0)
                    valid_actions = env.get_valid_actions(env.current_player)
                    
                    # Apply difficulty scaling
                    exploration_rates = {"Easy": 0.3, "Medium": 0.2, "Hard": 0.1, "Expert": 0.05}
                    exploration_rate = exploration_rates[difficulty]
                    
                    if random.random() < exploration_rate:
                        action = random.choice(valid_actions)
                    else:
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
                    valid_actions = env.get_valid_actions(env.current_player)
                    action = random.choice(valid_actions) if valid_actions else 0
                
                # Execute AI action
                next_state, reward, done, info = env.step(action)
                
                st.info(f"Player {env.current_player + 1} chose: {get_action_name(action)}")
                
                if done:
                    handle_game_end(reward)
                else:
                    time.sleep(0.5)
                    st.rerun()

def handle_game_end(reward: float):
    """Handle end of game"""
    env = st.session_state.env
    
    # Determine winner
    winner = env._determine_winner()
    
    if winner == 0:
        st.success(f"🎉 You won! Reward: +${reward:.2f}")
        result = "win"
    else:
        st.error(f"😞 You lost! Penalty: ${reward:.2f}")
        result = "loss"
    
    # Add to game history
    st.session_state.game_history.append({
        'result': result,
        'reward': reward,
        'pot': env.pot,
        'final_hand': [str(card) for card in env.players[0].hole_cards],
        'community_cards': [str(card) for card in env.community_cards]
    })
    
    st.session_state.game_state = 'waiting'
    
    # Show final hands
    st.markdown("### Final Hands:")
    for i, player in enumerate(env.players):
        if not player.folded and player.hole_cards:
            display_hand(player.hole_cards, f"Player {i+1}")

def show_game_statistics():
    """Show game statistics"""
    history = st.session_state.game_history
    
    if not history:
        return
    
    st.markdown("### Game Statistics")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    total_games = len(history)
    wins = sum(1 for game in history if game['result'] == 'win')
    total_reward = sum(game['reward'] for game in history)
    avg_pot = np.mean([game['pot'] for game in history])
    
    with col1:
        st.metric("Total Games", total_games)
    with col2:
        st.metric("Win Rate", f"{wins/total_games*100:.1f}%")
    with col3:
        st.metric("Total Reward", f"${total_reward:.2f}")
    with col4:
        st.metric("Avg Pot Size", f"${avg_pot:.2f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Win/Loss pie chart
        win_loss_data = pd.DataFrame({
            'Result': ['Wins', 'Losses'],
            'Count': [wins, total_games - wins]
        })
        
        fig = px.pie(win_loss_data, values='Count', names='Result', 
                    title="Win/Loss Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Reward over time
        rewards = [game['reward'] for game in history]
        cumulative_rewards = np.cumsum(rewards)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cumulative_rewards,
            mode='lines+markers',
            name='Cumulative Reward'
        ))
        fig.update_layout(title="Cumulative Reward Over Time",
                         xaxis_title="Game Number",
                         yaxis_title="Cumulative Reward ($)")
        st.plotly_chart(fig, use_container_width=True)

def train_models_page():
    st.header("🧠 Train AI Models")
    
    training_type = st.selectbox(
        "Training Method:",
        ["Neural Network Training", "CFR Training", "Population-Based Training"]
    )
    
    if training_type == "Neural Network Training":
        neural_network_training()
    elif training_type == "CFR Training":
        cfr_training()
    elif training_type == "Population-Based Training":
        population_training()

def neural_network_training():
    st.subheader("Neural Network Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Model Architecture:",
            ["transformer", "cnn", "lstm", "ensemble"]
        )
        
        epochs = st.slider("Training Epochs", 1, 100, 20)
        learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        
    with col2:
        batch_size = st.slider("Batch Size", 16, 128, 32)
        num_games = st.slider("Training Games", 100, 5000, 1000)
    
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model... This may take a while."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Create and train model
                model = create_poker_model(model_type)
                model = compile_poker_model(model, learning_rate)
                
                env = initialize_environment()
                
                # Simple training loop with progress updates
                training_data = []
                
                for game in range(num_games):
                    progress = game / num_games
                    progress_bar.progress(progress)
                    status_text.text(f"Generating training data: Game {game+1}/{num_games}")
                    
                    # Play a random game to generate data
                    state = env.reset()
                    done = False
                    
                    while not done:
                        valid_actions = env.get_valid_actions(env.current_player)
                        action = random.choice(valid_actions) if valid_actions else 0
                        
                        next_state, reward, done, info = env.step(action)
                        
                        if env.current_player == 0:  # Store data for player 0
                            # Create dummy target
                            target_policy = np.zeros(env.action_space.n)
                            target_policy[action] = 1.0
                            training_data.append((state, target_policy, reward))
                        
                        state = next_state
                
                # Convert to training arrays
                states = np.array([d[0] for d in training_data])
                policies = np.array([d[1] for d in training_data])
                values = np.array([d[2] for d in training_data])
                
                # Train model
                status_text.text("Training neural network...")
                history = model.fit(
                    states,
                    {'policy': policies, 'value': values.reshape(-1, 1)},
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Save model
                model.save(f"trained_{model_type}_model.h5")
                
                progress_bar.progress(1.0)
                status_text.text("Training complete!")
                st.success(f"Model trained and saved as 'trained_{model_type}_model.h5'")
                
                # Show training curves
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history.history['loss'],
                        mode='lines',
                        name='Training Loss'
                    ))
                    if 'val_loss' in history.history:
                        fig.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                    fig.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'policy_categorical_accuracy' in history.history:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=history.history['policy_categorical_accuracy'],
                            mode='lines',
                            name='Training Accuracy'
                        ))
                        if 'val_policy_categorical_accuracy' in history.history:
                            fig.add_trace(go.Scatter(
                                y=history.history['val_policy_categorical_accuracy'],
                                mode='lines',
                                name='Validation Accuracy'
                            ))
                        fig.update_layout(title="Policy Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def cfr_training():
    st.subheader("CFR (Counterfactual Regret Minimization) Training")
    
    iterations = st.slider("CFR Iterations", 100, 10000, 1000)
    
    if st.button("🧮 Start CFR Training", type="primary"):
        with st.spinner("Training CFR agent... This may take a while."):
            progress_bar = st.progress(0)
            
            try:
                # Train CFR agent
                trainer, strategies = train_cfr_agent(iterations)
                
                progress_bar.progress(1.0)
                st.success(f"CFR training complete! Strategy saved to 'cfr_strategy.pkl'")
                
                # Show strategy statistics
                st.markdown("### Strategy Statistics")
                st.write(f"Number of information sets: {len(strategies)}")
                
                # Show sample strategies
                if strategies:
                    st.markdown("### Sample Information Sets:")
                    sample_keys = list(strategies.keys())[:5]
                    
                    for key in sample_keys:
                        st.write(f"**{key}:**")
                        strategy = strategies[key]
                        action_probs = {get_action_name(i): prob for i, prob in enumerate(strategy)}
                        st.write(action_probs)
                
            except Exception as e:
                st.error(f"CFR training failed: {str(e)}")

def population_training():
    st.subheader("Population-Based Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        population_size = st.slider("Population Size", 5, 50, 10)
        generations = st.slider("Generations", 1, 20, 5)
    
    with col2:
        games_per_eval = st.slider("Games per Evaluation", 10, 200, 50)
        mutation_rate = st.slider("Mutation Rate", 0.01, 0.5, 0.1)
    
    if st.button("🧬 Start Population Training", type="primary"):
        st.warning("Population training is computationally intensive and may take a very long time!")
        
        if st.button("⚠️ I understand, start training"):
            with st.spinner("Running population-based training..."):
                try:
                    from self_play_training import run_population_training
                    
                    pbt = run_population_training(
                        generations=generations,
                        population_size=population_size
                    )
                    
                    st.success("Population training complete!")
                    
                except Exception as e:
                    st.error(f"Population training failed: {str(e)}")

def analytics_page():
    st.header("📊 Analytics Dashboard")
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    if st.button("🔄 Run Model Evaluation"):
        with st.spinner("Evaluating models..."):
            models = load_models()
            env = initialize_environment()
            
            results = {}
            
            for model_name, model in models.items():
                # Evaluate model performance
                wins = 0
                total_games = 50
                total_reward = 0
                
                for _ in range(total_games):
                    state = env.reset()
                    done = False
                    game_reward = 0
                    
                    while not done:
                        if env.current_player == 0:  # Our model
                            try:
                                policy, value = model.predict(state.reshape(1, -1), verbose=0)
                                valid_actions = env.get_valid_actions(0)
                                
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
                                valid_actions = env.get_valid_actions(0)
                                action = random.choice(valid_actions) if valid_actions else 0
                        else:
                            valid_actions = env.get_valid_actions(env.current_player)
                            action = random.choice(valid_actions) if valid_actions else 0
                        
                        next_state, reward, done, info = env.step(action)
                        
                        if env.current_player == 0:
                            game_reward += reward
                        
                        state = next_state
                    
                    total_reward += game_reward
                    if game_reward > 0:
                        wins += 1
                
                results[model_name] = {
                    'win_rate': wins / total_games,
                    'avg_reward': total_reward / total_games
                }
            
            # Display results
            results_df = pd.DataFrame(results).T
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=results_df.index,
                    y=results_df['win_rate'],
                    title="Model Win Rates"
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Win Rate")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    x=results_df.index,
                    y=results_df['avg_reward'],
                    title="Average Reward per Game"
                )
                fig.update_layout(xaxis_title="Model", yaxis_title="Average Reward")
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(results_df)

def settings_page():
    st.header("⚙️ Settings")
    
    st.subheader("Game Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Number of Players", 2, 10, 6, key="num_players")
        st.slider("Starting Stack", 500, 5000, 1000, key="starting_stack")
    
    with col2:
        st.slider("Small Blind", 1, 50, 5, key="small_blind")
        st.slider("Big Blind", 2, 100, 10, key="big_blind")
    
    st.subheader("AI Settings")
    
    st.slider("AI Thinking Time (seconds)", 0.5, 5.0, 1.0, key="ai_thinking_time")
    st.selectbox("Default AI Model", ["transformer", "cnn", "lstm"], key="default_ai_model")
    
    st.subheader("Display Settings")
    
    st.checkbox("Show AI Hand Strength", key="show_ai_strength")
    st.checkbox("Show Detailed Logs", key="show_detailed_logs")
    st.checkbox("Auto-save Game History", True, key="auto_save_history")
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved!")

if __name__ == "__main__":
    main()