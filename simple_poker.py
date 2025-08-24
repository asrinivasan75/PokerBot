#!/usr/bin/env python3

import random
from enum import Enum

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank}{self.suit}"

class Action(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3

class SimplePoker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.player_cards = self.deal_hand()
        self.ai_cards = self.deal_hand()
        self.community_cards = []
        self.pot = 20  # Starting pot
        self.player_bet = 10
        self.ai_bet = 10
        self.round = 0  # 0=pre-flop, 1=flop, 2=turn, 3=river
        self.game_over = False
        self.winner = None
        
        return self.get_state()
    
    def deal_hand(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        return [Card(random.choice(ranks), random.choice(suits)) for _ in range(2)]
    
    def deal_flop(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        return [Card(random.choice(ranks), random.choice(suits)) for _ in range(3)]
    
    def deal_card(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        return Card(random.choice(ranks), random.choice(suits))
    
    def get_state(self):
        return {
            'player_cards': [str(card) for card in self.player_cards],
            'community_cards': [str(card) for card in self.community_cards],
            'pot': self.pot,
            'player_bet': self.player_bet,
            'ai_bet': self.ai_bet,
            'round': self.round,
            'game_over': self.game_over,
            'winner': self.winner
        }
    
    def player_action(self, action):
        if self.game_over:
            return self.get_state(), "Game is over"
        
        if action == Action.FOLD.value:
            self.game_over = True
            self.winner = "AI"
            return self.get_state(), "You folded. AI wins!"
        
        elif action == Action.CHECK.value:
            pass  # No bet
        
        elif action == Action.CALL.value:
            call_amount = self.ai_bet - self.player_bet
            self.player_bet += call_amount
            self.pot += call_amount
        
        elif action == Action.BET.value:
            bet_amount = 20
            self.player_bet += bet_amount
            self.pot += bet_amount
        
        # AI's turn
        ai_action = self.get_ai_action()
        message = self.handle_ai_action(ai_action)
        
        # Advance round if both players are done
        if not self.game_over and self.player_bet == self.ai_bet:
            self.advance_round()
        
        return self.get_state(), message
    
    def get_ai_action(self):
        # Simple AI logic
        actions = [Action.FOLD, Action.CHECK, Action.CALL, Action.BET]
        weights = [0.1, 0.3, 0.4, 0.2]  # AI is slightly aggressive
        return random.choices(actions, weights=weights)[0]
    
    def handle_ai_action(self, action):
        if action == Action.FOLD:
            self.game_over = True
            self.winner = "Player"
            return "AI folded. You win!"
        
        elif action == Action.CHECK:
            return "AI checked."
        
        elif action == Action.CALL:
            call_amount = self.player_bet - self.ai_bet
            self.ai_bet += call_amount
            self.pot += call_amount
            return f"AI called (${call_amount})."
        
        elif action == Action.BET:
            bet_amount = 20
            self.ai_bet += bet_amount
            self.pot += bet_amount
            return f"AI bet ${bet_amount}."
    
    def advance_round(self):
        self.round += 1
        
        if self.round == 1:  # Flop
            self.community_cards = self.deal_flop()
        elif self.round == 2:  # Turn
            self.community_cards.append(self.deal_card())
        elif self.round == 3:  # River
            self.community_cards.append(self.deal_card())
        elif self.round == 4:  # Showdown
            self.game_over = True
            self.winner = self.determine_winner()
    
    def determine_winner(self):
        # Simple random winner for now
        return random.choice(["Player", "AI"])

def main():
    print("🃏 Simple Poker Game")
    print("Actions: 0=Fold, 1=Check, 2=Call, 3=Bet")
    print("-" * 40)
    
    game = SimplePoker()
    state = game.reset()
    
    while not state['game_over']:
        print(f"\n--- Round {state['round'] + 1} ---")
        print(f"Your cards: {state['player_cards']}")
        print(f"Community: {state['community_cards']}")
        print(f"Pot: ${state['pot']}")
        print(f"Your bet: ${state['player_bet']} | AI bet: ${state['ai_bet']}")
        
        try:
            action = int(input("\nYour action (0-3): "))
            if action not in [0, 1, 2, 3]:
                print("Invalid action!")
                continue
            
            state, message = game.player_action(action)
            print(f">>> {message}")
            
        except (ValueError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
    
    if state['game_over']:
        print(f"\n🎉 Game Over! Winner: {state['winner']}")
        print(f"Final pot: ${state['pot']}")
        print(f"Your cards: {state['player_cards']}")
        print(f"Community: {state['community_cards']}")

if __name__ == "__main__":
    main()