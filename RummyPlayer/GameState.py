class GameState:
    def __init__(self):
        self.hand = []          # Current cards in hand
        self.discard_pile = []  # Track discard pile
        self.meld_piles = {}    # Track all melds in play
        self.opponent_name = "" # Store opponent's name
        self.game_id = ""       # Current game ID
        self.scores = {}        # Track game scores
        self.seen_cards = set() # Track all cards we've seen

    def start_new_game(self, game_id, opponent):
        self.__init__()  # Reset all state
        self.game_id = game_id
        self.opponent_name = opponent

    def set_hand(self, cards):
        self.hand = cards
        self.hand.sort()
        for card in cards:
            self.seen_cards.add(card)

    def add_card_to_hand(self, card):
        self.hand.append(card)
        self.hand.sort()
        self.seen_cards.add(card)

    def remove_card_from_hand(self, card):
        self.hand.remove(card)

    def update_discard(self, card, is_adding):
        if is_adding:
            self.discard_pile.insert(0, card)
            self.seen_cards.add(card)
        else:
            if self.discard_pile:
                self.discard_pile.pop(0)

    def update_meld(self, meld_number, cards):
        self.meld_piles[meld_number] = cards
        for card in cards:
            self.seen_cards.add(card)

    def get_potential_melds(self):
        return {
            'sets': self.find_sets(),
            'runs': self.find_runs()
        }

    def find_sets(self):
        # Group cards by rank
        ranks = {}
        for card in self.hand:
            rank = card[0]  # First character is rank
            if rank not in ranks:
                ranks[rank] = []
            ranks[rank].append(card)
        
        # Return groups of 3 or more
        return [cards for cards in ranks.values() if len(cards) >= 3]

    def find_runs(self):
        # Group cards by suit
        suits = {}
        for card in self.hand:
            suit = card[1]  # Second character is suit
            if suit not in suits:
                suits[suit] = []
            suits[suit].append(card)
        
        # Find sequences of 3 or more
        runs = []
        for suit_cards in suits.values():
            # Sort by rank using custom order
            rank_order = '23456789TJQKA'
            sorted_cards = sorted(suit_cards, key=lambda card: rank_order.index(card[0]))
            
            # Find consecutive sequences
            current_run = []
            for card in sorted_cards:
                if not current_run or self._is_next_in_sequence(current_run[-1], card):
                    current_run.append(card)
                else:
                    if len(current_run) >= 3:
                        runs.append(current_run[:])
                    current_run = [card]
            if len(current_run) >= 3:
                runs.append(current_run)
        return runs

    def _is_next_in_sequence(self, card1, card2):
        rank_order = '23456789TJQKA'
        rank1_idx = rank_order.index(card1[0])
        rank2_idx = rank_order.index(card2[0])
        return rank2_idx == rank1_idx + 1

    def should_take_discard(self):
        if not self.discard_pile:
            return False
            
        top_card = self.discard_pile[0]
        
        # Check if completes a set
        rank_matches = sum(1 for card in self.hand if card[0] == top_card[0])
        if rank_matches >= 2:
            return True
            
        # Check if could complete a run
        suit_cards = [card for card in self.hand if card[1] == top_card[1]]
        if len(suit_cards) >= 2:
            # Add top card to suit cards and see if it forms a run
            test_cards = suit_cards + [top_card]
            rank_order = '23456789TJQKA'
            test_cards.sort(key=lambda card: rank_order.index(card[0]))
            
            # Check for consecutive cards
            for i in range(len(test_cards) - 2):
                if (self._is_next_in_sequence(test_cards[i], test_cards[i+1]) and
                    self._is_next_in_sequence(test_cards[i+1], test_cards[i+2])):
                    return True
        
        return False

    def calculate_hand_value(self):
        #Calculate the point value of cards in hand
        value_map = {'T':10, 'J':10, 'Q':10, 'K':10, 'A':10}
        value = 0
        for card in self.hand:
            rank = card[0]
            #print(rank)
            if rank in value_map:
                value += value_map[rank]
            else:
                value += int(rank)
        return value

    def card_value(self, card):
        """Get the point value of a single card"""
        value_map = {'T':10, 'J':10, 'Q':10, 'K':10, 'A':10}
        return value_map.get(card[0], int(card[0]))

    def find_best_meld_combination(self):
        """Find the best combination of melds to play"""
        # Get all possible melds
        sets = self.find_sets()
        runs = self.find_runs()
        
        # Score each meld based on points
        meld_scores = []
        
        # Score sets
        for meld in sets:
            score = sum(self.card_value(card) for card in meld)
            meld_scores.append((meld, score))
        
        # Score runs
        for meld in runs:
            score = sum(self.card_value(card) for card in meld)
            meld_scores.append((meld, score))
        
        # Sort by score, highest first
        return sorted(meld_scores, key=lambda x: x[1], reverse=True)

    def can_lay_off_on_meld(self, card, meld_cards):
        """Check if a card can be laid off on an existing meld"""
        if not meld_cards:
            return False
            
        # Check if it's a set (same rank)
        if all(c[0] == meld_cards[0][0] for c in meld_cards):
            return card[0] == meld_cards[0][0]
            
        # Check if it's a run (same suit, sequential)
        if all(c[1] == meld_cards[0][1] for c in meld_cards):
            rank_order = '23456789TJQKA'
            meld_ranks = [rank_order.index(c[0]) for c in meld_cards]
            card_rank = rank_order.index(card[0])
            return (card[1] == meld_cards[0][1] and 
                   (card_rank == min(meld_ranks) - 1 or 
                    card_rank == max(meld_ranks) + 1))
        
        return False

    def validate_play(self, card, play_type, meld_num=None):
        """Validate if a play is legal
        Args:
            card (str): Card to validate
            play_type (str): Type of play ('meld', 'layoff', 'discard')
            meld_num (int, optional): Meld number for layoffs
        Returns:
            bool: True if play is legal, False otherwise
        """
        # Check if card is in hand
        if card not in self.hand:
            return False
            
        if play_type == 'layoff':
            # Check if meld exists
            if meld_num not in self.meld_piles:
                return False
            # Check if card can be laid off
            return self.can_lay_off_on_meld(card, self.meld_piles[meld_num])
            
        return True

    def find_best_discard(self):
        """Choose the best card to discard based on various factors"""
        if not self.hand:
            return None
            
        # Calculate risk score for each card
        card_scores = []
        for card in self.hand:
            score = 0
            # Higher score = better to keep
            
            # Check if part of a potential set
            rank_matches = sum(1 for c in self.hand if c[0] == card[0])
            score += rank_matches * 10
            
            # Check if part of a potential run
            suit_cards = [c for c in self.hand if c[1] == card[1]]
            rank_order = '23456789TJQKA'
            suit_cards.sort(key=lambda x: rank_order.index(x[0]))
            
            # Find consecutive cards
            for i, c in enumerate(suit_cards):
                if c == card:
                    # Look at up to 2 cards before and after
                    for j in range(max(0, i-2), min(len(suit_cards), i+3)):
                        if j != i:
                            diff = abs(rank_order.index(suit_cards[j][0]) - rank_order.index(card[0]))
                            if diff <= 2:  # Within 2 positions
                                score += (3 - diff) * 5  # More points for closer cards
            
            # Consider card value (prefer discarding high-value cards if not part of potential meld)
            if score < 10:  # If not strongly connected to other cards
                score -= self.card_value(card)
            
            card_scores.append((card, score))
        
        # Return card with lowest score (worst to keep)
        return min(card_scores, key=lambda x: x[1])[0]

    def find_layoff_opportunities(self):
        """Find all possible layoff opportunities for cards in hand"""
        layoffs = []
        for card in self.hand:
            for meld_num, meld_cards in self.meld_piles.items():
                if self.can_lay_off_on_meld(card, meld_cards):
                    layoffs.append((card, meld_num))
        return layoffs
