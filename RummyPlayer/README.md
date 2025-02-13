# Rummy Player API

This is my implementation of a Rummy card game player for CSCI-4332 Artificial Intelligence at Hardin-Simmons University.

## Features

### Game State Management
- Tracks current hand, discard pile, and meld piles
- Maintains game scores and seen cards
- Handles multiple opponents and game events

### Strategic Decision Making
1. **Meld Selection**
   - Identifies both sets and runs
   - Scores potential melds based on point value
   - Prioritizes higher-scoring combinations

2. **Drawing Strategy**
   - Evaluates discard pile for potential melds
   - Considers both sets and runs when deciding
   - Takes into account seen cards

3. **Discard Strategy**
   - Evaluates cards based on meld potential
   - Considers both current and potential future melds
   - Prefers keeping cards that could form runs or sets

4. **Layoff Opportunities**
   - Identifies chances to lay off on existing melds
   - Handles both sets and runs
   - Prioritizes high-value layoffs

## Installation

```bash
pip install -r requirements.txt
```

## Running the Game

1. Start the server:
```bash
python3 main.py
```

2. Run tests:
```bash
python3 test_gamestate.py
```

## Implementation Details

### GameState Class
- Manages game state and decision-making logic
- Implements card evaluation and meld finding algorithms
- Handles game events and updates

### API Endpoints
- `/start-2p-game/`: Initializes a new game
- `/start-2p-hand/`: Starts a new hand
- `/draw/`: Handles drawing cards
- `/lay-down/`: Manages playing melds and discarding

## Testing
Comprehensive test suite includes:
- Unit tests for all game logic
- API endpoint testing
- Strategy validation

## Author
Jonathan Makenene
