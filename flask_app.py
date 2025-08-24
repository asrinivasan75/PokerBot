from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import tensorflow as tf
import numpy as np
import json
import random
import uuid
from datetime import datetime
import os
from advanced_poker_env import AdvancedPokerEnv, Action, PokerCard
from neural_networks import create_poker_model, compile_poker_model
from cfr_algorithm import CFRAgent

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global storage for games and models
games = {}
models = {}

def load_poker_models():
    """Load available poker models - simplified to avoid blocking"""
    global models
    
    # Simplified - just create placeholders
    models = {"simple_ai": "placeholder"}
    print(f"Loaded {len(models)} poker models (simplified mode)")

class GameRoom:
    """Represents a poker game room"""
    
    def __init__(self, room_id: str, max_players: int = 6):
        self.room_id = room_id
        self.max_players = max_players
        self.players = {}  # player_id -> player_info
        self.env = AdvancedPokerEnv(num_players=max_players)
        self.game_state = 'waiting'  # waiting, playing, finished
        self.current_hand = 0
        self.ai_models = ['transformer']  # AI models in this game
        self.game_history = []
        
    def add_player(self, player_id: str, player_name: str, is_ai: bool = False):
        """Add a player to the game"""
        if len(self.players) >= self.max_players:
            return False
        
        self.players[player_id] = {
            'id': player_id,
            'name': player_name,
            'is_ai': is_ai,
            'stack': 1000,
            'position': len(self.players),
            'connected': True
        }
        return True
    
    def remove_player(self, player_id: str):
        """Remove a player from the game"""
        if player_id in self.players:
            del self.players[player_id]
    
    def start_game(self):
        """Start the poker game"""
        if len(self.players) < 2:
            return False
        
        self.game_state = 'playing'
        self.env.reset()
        return True
    
    def get_game_state(self, player_id: str = None):
        """Get current game state for frontend"""
        state = {
            'room_id': self.room_id,
            'game_state': self.game_state,
            'players': list(self.players.values()),
            'pot': self.env.pot,
            'current_bet': self.env.current_bet,
            'game_phase': self.env.game_phase.name,
            'community_cards': [self._card_to_dict(card) for card in self.env.community_cards],
            'current_player': self.env.current_player,
            'hand_number': self.current_hand
        }
        
        # Add player-specific information
        if player_id and player_id in self.players:
            player_pos = self.players[player_id]['position']
            if player_pos < len(self.env.players):
                env_player = self.env.players[player_pos]
                state['player_cards'] = [self._card_to_dict(card) for card in env_player.hole_cards]
                state['player_stack'] = env_player.stack
                state['player_current_bet'] = env_player.current_bet
                state['valid_actions'] = self.env.get_valid_actions(player_pos)
        
        return state
    
    def _card_to_dict(self, card: PokerCard):
        """Convert PokerCard to dictionary"""
        return {
            'rank': card.rank,
            'suit': card.suit,
            'display': str(card)
        }
    
    def execute_action(self, player_id: str, action: int):
        """Execute a player action"""
        if player_id not in self.players:
            return False, "Player not in game"
        
        player_pos = self.players[player_id]['position']
        
        if self.env.current_player != player_pos:
            return False, "Not your turn"
        
        # Validate and execute action
        valid_actions = self.env.get_valid_actions(player_pos)
        if action not in valid_actions:
            return False, "Invalid action"
        
        try:
            next_state, reward, done, info = self.env.step(action)
            
            # Log the action
            action_log = {
                'player_id': player_id,
                'player_name': self.players[player_id]['name'],
                'action': Action(action).name,
                'timestamp': datetime.now().isoformat()
            }
            self.game_history.append(action_log)
            
            return True, "Action executed successfully"
        except Exception as e:
            return False, f"Error executing action: {str(e)}"

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/models')
def get_models():
    """Get available AI models"""
    return jsonify(list(models.keys()))

@app.route('/api/create_room', methods=['POST'])
def create_room():
    """Create a new game room"""
    data = request.json
    room_id = str(uuid.uuid4())[:8]
    max_players = data.get('max_players', 6)
    
    games[room_id] = GameRoom(room_id, max_players)
    
    return jsonify({
        'room_id': room_id,
        'success': True
    })

@app.route('/api/join_room', methods=['POST'])
def join_room_api():
    """Join a game room"""
    data = request.json
    room_id = data.get('room_id')
    player_name = data.get('player_name', 'Anonymous')
    
    if room_id not in games:
        return jsonify({'success': False, 'error': 'Room not found'})
    
    player_id = str(uuid.uuid4())
    game = games[room_id]
    
    if game.add_player(player_id, player_name):
        return jsonify({
            'success': True,
            'player_id': player_id,
            'game_state': game.get_game_state(player_id)
        })
    else:
        return jsonify({'success': False, 'error': 'Room is full'})

@app.route('/api/game_state/<room_id>/<player_id>')
def get_game_state(room_id, player_id):
    """Get current game state"""
    if room_id not in games:
        return jsonify({'error': 'Room not found'}), 404
    
    game = games[room_id]
    return jsonify(game.get_game_state(player_id))

@app.route('/api/action', methods=['POST'])
def execute_action():
    """Execute a player action"""
    data = request.json
    room_id = data.get('room_id')
    player_id = data.get('player_id')
    action = data.get('action')
    
    if room_id not in games:
        return jsonify({'success': False, 'error': 'Room not found'})
    
    game = games[room_id]
    success, message = game.execute_action(player_id, action)
    
    return jsonify({
        'success': success,
        'message': message,
        'game_state': game.get_game_state(player_id)
    })

# Socket.IO events for real-time updates
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('join_game')
def handle_join_game(data):
    """Handle player joining a game room"""
    room_id = data.get('room_id')
    player_id = data.get('player_id')
    
    if room_id in games:
        join_room(room_id)
        game = games[room_id]
        
        # Broadcast updated game state to all players in room
        emit('game_update', game.get_game_state(), room=room_id)
        
        emit('joined_game', {
            'success': True,
            'room_id': room_id,
            'game_state': game.get_game_state(player_id)
        })
    else:
        emit('joined_game', {'success': False, 'error': 'Room not found'})

@socketio.on('leave_game')
def handle_leave_game(data):
    """Handle player leaving a game room"""
    room_id = data.get('room_id')
    player_id = data.get('player_id')
    
    if room_id in games:
        leave_room(room_id)
        game = games[room_id]
        game.remove_player(player_id)
        
        # Broadcast updated game state
        emit('game_update', game.get_game_state(), room=room_id)

@socketio.on('player_action')
def handle_player_action(data):
    """Handle player action via WebSocket"""
    room_id = data.get('room_id')
    player_id = data.get('player_id')
    action = data.get('action')
    
    if room_id in games:
        game = games[room_id]
        success, message = game.execute_action(player_id, action)
        
        if success:
            # Broadcast game update to all players
            emit('game_update', game.get_game_state(), room=room_id)
            emit('action_result', {'success': True, 'message': message})
            
            # Handle AI turns
            handle_ai_turns(room_id)
        else:
            emit('action_result', {'success': False, 'message': message})

def handle_ai_turns(room_id: str):
    """Handle AI player turns"""
    if room_id not in games:
        return
    
    game = games[room_id]
    
    # Process AI turns
    while (game.env.current_player < len(game.env.players) and 
           game.env.current_player < len(list(game.players.values()))):
        
        current_player_info = list(game.players.values())[game.env.current_player]
        
        if current_player_info.get('is_ai', False):
            # AI player's turn
            ai_action = get_ai_action(game)
            
            if ai_action is not None:
                success, message = game.execute_action(
                    current_player_info['id'], 
                    ai_action
                )
                
                if success:
                    # Broadcast AI action
                    socketio.emit('ai_action', {
                        'player_name': current_player_info['name'],
                        'action': Action(ai_action).name,
                        'game_state': game.get_game_state()
                    }, room=room_id)
                else:
                    break
            else:
                break
        else:
            break  # Human player's turn

def get_ai_action(game: GameRoom) -> int:
    """Get AI action using simple random strategy"""
    try:
        current_player = game.env.current_player
        valid_actions = game.env.get_valid_actions(current_player)
        
        if not valid_actions:
            return 0  # Fold
        
        # Simple random AI for now
        return random.choice(valid_actions)
        
    except Exception as e:
        print(f"Error getting AI action: {e}")
        valid_actions = game.env.get_valid_actions(game.env.current_player)
        return random.choice(valid_actions) if valid_actions else 0

# HTML Template
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(template_dir, exist_ok=True)

@app.route('/create_templates')
def create_templates():
    """Create HTML templates"""
    
    # Create index.html
    index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced PokerBot</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.3.4/socket.io.js"></script>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .poker-table {
            background: #0f5132;
            border-radius: 50%;
            width: 600px;
            height: 400px;
            margin: 0 auto;
            position: relative;
            border: 8px solid #8B4513;
        }
        
        .player-seat {
            position: absolute;
            width: 100px;
            height: 80px;
            background: #fff;
            color: #000;
            border-radius: 10px;
            padding: 10px;
            text-align: center;
            font-size: 12px;
        }
        
        .community-cards {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            display: flex;
            gap: 5px;
        }
        
        .card {
            width: 40px;
            height: 60px;
            background: white;
            color: black;
            border: 1px solid #333;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 12px;
        }
        
        .red-card { color: red; }
        .black-card { color: black; }
        
        .controls {
            margin-top: 30px;
            text-align: center;
        }
        
        .btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        .btn:hover {
            background: #0056b3;
        }
        
        .btn-danger {
            background: #dc3545;
        }
        
        .btn-danger:hover {
            background: #c82333;
        }
        
        .btn-success {
            background: #28a745;
        }
        
        .btn-success:hover {
            background: #218838;
        }
        
        .game-info {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        
        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }
        
        .modal-content {
            background-color: #fefefe;
            color: black;
            margin: 15% auto;
            padding: 20px;
            border-radius: 10px;
            width: 400px;
        }
        
        .input-group {
            margin: 15px 0;
        }
        
        .input-group label {
            display: block;
            margin-bottom: 5px;
        }
        
        .input-group input, .input-group select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect } = React;
        
        const socket = io();
        
        function PokerApp() {
            const [gameState, setGameState] = useState(null);
            const [playerId, setPlayerId] = useState(null);
            const [roomId, setRoomId] = useState(null);
            const [playerName, setPlayerName] = useState('');
            const [showJoinModal, setShowJoinModal] = useState(false);
            const [gameHistory, setGameHistory] = useState([]);
            
            useEffect(() => {
                socket.on('game_update', (state) => {
                    setGameState(state);
                });
                
                socket.on('joined_game', (data) => {
                    if (data.success) {
                        setGameState(data.game_state);
                        setRoomId(data.room_id);
                        setShowJoinModal(false);
                    } else {
                        alert(data.error);
                    }
                });
                
                socket.on('ai_action', (data) => {
                    setGameHistory(prev => [...prev, `${data.player_name} chose ${data.action}`]);
                });
                
                socket.on('action_result', (data) => {
                    if (!data.success) {
                        alert(data.message);
                    }
                });
                
                return () => {
                    socket.off('game_update');
                    socket.off('joined_game');
                    socket.off('ai_action');
                    socket.off('action_result');
                };
            }, []);
            
            const createRoom = async () => {
                try {
                    const response = await fetch('/api/create_room', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ max_players: 6 }),
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        setRoomId(data.room_id);
                        setShowJoinModal(true);
                    }
                } catch (error) {
                    console.error('Error creating room:', error);
                }
            };
            
            const joinRoom = async () => {
                if (!playerName.trim()) {
                    alert('Please enter your name');
                    return;
                }
                
                try {
                    const response = await fetch('/api/join_room', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            room_id: roomId,
                            player_name: playerName,
                        }),
                    });
                    
                    const data = await response.json();
                    if (data.success) {
                        setPlayerId(data.player_id);
                        socket.emit('join_game', {
                            room_id: roomId,
                            player_id: data.player_id,
                        });
                    } else {
                        alert(data.error);
                    }
                } catch (error) {
                    console.error('Error joining room:', error);
                }
            };
            
            const executeAction = (action) => {
                if (playerId && roomId) {
                    socket.emit('player_action', {
                        room_id: roomId,
                        player_id: playerId,
                        action: action,
                    });
                }
            };
            
            const renderCard = (card) => {
                if (!card) return <div className="card">?</div>;
                
                const isRed = card.suit === 'h' || card.suit === 'd';
                return (
                    <div className={`card ${isRed ? 'red-card' : 'black-card'}`}>
                        {card.display}
                    </div>
                );
            };
            
            const getActionName = (actionId) => {
                const actions = {
                    0: 'Fold',
                    1: 'Check',
                    2: 'Call',
                    3: 'Bet 25%',
                    4: 'Bet 50%',
                    5: 'Bet 75%',
                    6: 'Bet 100%',
                    7: 'All In'
                };
                return actions[actionId] || `Action ${actionId}`;
            };
            
            if (!gameState) {
                return (
                    <div className="container">
                        <div className="header">
                            <h1>🃏 Advanced PokerBot</h1>
                            <p>Play poker against advanced AI opponents</p>
                        </div>
                        
                        <div style={{textAlign: 'center'}}>
                            <button className="btn" onClick={createRoom}>
                                Create New Game
                            </button>
                        </div>
                        
                        {showJoinModal && (
                            <div className="modal" style={{display: 'block'}}>
                                <div className="modal-content">
                                    <h2>Join Game</h2>
                                    <div className="input-group">
                                        <label>Your Name:</label>
                                        <input
                                            type="text"
                                            value={playerName}
                                            onChange={(e) => setPlayerName(e.target.value)}
                                            placeholder="Enter your name"
                                        />
                                    </div>
                                    <div className="input-group">
                                        <label>Room ID:</label>
                                        <input
                                            type="text"
                                            value={roomId}
                                            readOnly
                                        />
                                    </div>
                                    <div style={{textAlign: 'center'}}>
                                        <button className="btn btn-success" onClick={joinRoom}>
                                            Join Game
                                        </button>
                                        <button className="btn" onClick={() => setShowJoinModal(false)}>
                                            Cancel
                                        </button>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                );
            }
            
            return (
                <div className="container">
                    <div className="header">
                        <h1>🃏 Poker Game - Room {roomId}</h1>
                    </div>
                    
                    <div className="game-info">
                        <div style={{display: 'flex', justifyContent: 'space-between'}}>
                            <div>
                                <strong>Pot: ${gameState.pot}</strong>
                            </div>
                            <div>
                                <strong>Phase: {gameState.game_phase}</strong>
                            </div>
                            <div>
                                <strong>Current Bet: ${gameState.current_bet}</strong>
                            </div>
                        </div>
                    </div>
                    
                    <div className="poker-table">
                        <div className="community-cards">
                            {gameState.community_cards.map((card, index) => (
                                <div key={index}>
                                    {renderCard(card)}
                                </div>
                            ))}
                        </div>
                        
                        {/* Position players around the table */}
                        {gameState.players.map((player, index) => (
                            <div 
                                key={player.id}
                                className="player-seat"
                                style={{
                                    top: `${20 + (index % 3) * 40}%`,
                                    left: `${10 + (Math.floor(index / 3)) * 80}%`
                                }}
                            >
                                <div><strong>{player.name}</strong></div>
                                <div>${player.stack || 1000}</div>
                                <div>{player.is_ai ? '🤖' : '👤'}</div>
                            </div>
                        ))}
                    </div>
                    
                    {gameState.player_cards && (
                        <div style={{textAlign: 'center', margin: '20px 0'}}>
                            <h3>Your Cards:</h3>
                            <div style={{display: 'flex', justifyContent: 'center', gap: '10px'}}>
                                {gameState.player_cards.map((card, index) => (
                                    <div key={index}>
                                        {renderCard(card)}
                                    </div>
                                ))}
                            </div>
                            <div style={{marginTop: '10px'}}>
                                <strong>Your Stack: ${gameState.player_stack}</strong> | 
                                <strong> Current Bet: ${gameState.player_current_bet}</strong>
                            </div>
                        </div>
                    )}
                    
                    {gameState.valid_actions && gameState.valid_actions.length > 0 && (
                        <div className="controls">
                            <h3>Your Turn - Choose an Action:</h3>
                            {gameState.valid_actions.map((action) => (
                                <button
                                    key={action}
                                    className={`btn ${action === 0 ? 'btn-danger' : action === 2 ? 'btn-success' : ''}`}
                                    onClick={() => executeAction(action)}
                                >
                                    {getActionName(action)}
                                </button>
                            ))}
                        </div>
                    )}
                    
                    {gameHistory.length > 0 && (
                        <div className="game-info">
                            <h3>Game History:</h3>
                            <div style={{maxHeight: '150px', overflowY: 'auto'}}>
                                {gameHistory.slice(-10).map((action, index) => (
                                    <div key={index}>{action}</div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            );
        }
        
        ReactDOM.render(<PokerApp />, document.getElementById('root'));
    </script>
</body>
</html>
    '''
    
    with open(os.path.join(template_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    return "Templates created successfully!"

if __name__ == '__main__':
    # Load models on startup
    load_poker_models()
    
    # Create templates
    create_templates()
    
    print("Starting Flask server...")
    print("Visit http://localhost:5000 to play poker!")
    
    socketio.run(app, debug=True, port=5000)