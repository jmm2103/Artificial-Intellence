import requests
from fastapi import FastAPI
import fastapi
from pydantic import BaseModel
import uvicorn
import os
import signal
import logging

"""
Rummy Player Implementation
By Jonathan Makenene

Version History:
1.0 - Initial API setup and basic functionality
1.1 - Added GameState class and basic strategy
1.2 - Implemented advanced meld selection and discard strategy
1.3 - Added layoff detection and improved hand evaluation
1.4 - Enhanced run detection and card scoring
1.5 - Added comprehensive testing and documentation

This implementation features:
- Smart meld selection (prioritizing high-value melds)
- Strategic discard choices based on potential melds
- Efficient layoff detection on existing melds
- Advanced run and set detection algorithms
"""

#port and username have been changed

DEBUG = True
PORT = 10800
USER_NAME = "Jonathan Makenene"
from GameState import GameState

# Initialize game state
game_state = GameState()

# set up the FastAPI application
app = FastAPI()
print("API connected")
# set up the API endpoints
@app.get("/")
async def root():
    ''' Root API simply confirms API is up and running.'''
    return {"status": "Running"}

# data class used to receive data from API POST
class GameInfo(BaseModel):
    game_id: str
    opponent: str
    hand: str

@app.post("/start-2p-game/")
async def start_game(game_info: GameInfo):
    ''' Game Server calls this endpoint to inform player a new game is starting. '''
    # TODO - Your code here - replace the lines below
    global game_state
    game_state.start_new_game(game_info.game_id, game_info.opponent)
    game_state.set_hand(game_info.hand.split(" "))
    logging.info("2p game started, hand is "+str(game_state.hand))
    return {"status": "OK"}

# data class used to receive data from API POST
class HandInfo(BaseModel):
    hand: str

@app.post("/start-2p-hand/")
async def start_hand(hand_info: HandInfo):
    ''' Game Server calls this endpoint to inform player a new hand is starting, continuing the previous game. '''
    # TODO - Your code here
    global game_state
    game_state.set_hand(hand_info.hand.split(" "))
    logging.info("2p hand started, hand is " + str(game_state.hand))
    return {"status": "OK"}

def process_events(event_text):
    ''' Shared function to process event text from various API endpoints '''
    # TODO - Your code here. Everything from here to end of function
    global game_state
    for event_line in event_text.splitlines():
        if ((USER_NAME + " draws") in event_line or (USER_NAME + " takes") in event_line):
            card = event_line.split(" ")[-1]
            game_state.add_card_to_hand(card)
            logging.info(f"Drew a {card}, hand is now: {game_state.hand}")
        if ("discards" in event_line):
            card = event_line.split(" ")[-1]
            game_state.update_discard(card, True)
        if ("takes" in event_line):
            game_state.update_discard("", False)
        if ("plays meld" in event_line):
            meld_info = event_line.split("meld(")[1]
            meld_num = int(meld_info.split(")")[0])
            cards = event_line.split(": ")[1].split(" ")
            game_state.update_meld(meld_num, cards)

# data class used to receive data from API POST
class UpdateInfo(BaseModel):
    game_id: str
    event: str

@app.post("/update-2p-game/")
async def update_2p_game(update_info: UpdateInfo):
    '''
        Game Server calls this endpoint to update player on game status and other players' moves.
        Typically only called at the end of game.
    '''
    # TODO - Your code here - update this section if you want
    process_events(update_info.event)
    return {"status": "OK"}

@app.post("/draw/")
async def draw(update_info: UpdateInfo):
    """Handle drawing decisions with the following strategy:
    1. If discard pile is empty, draw from stock
    2. Evaluate top discard card for:
       - Completing a set (needs 2 matching cards)
       - Completing a run (needs sequential cards)
       - High point value if no melds possible
    3. Consider opponent's discards and visible melds
    """
    global game_state
    process_events(update_info.event)
    
    # If discard pile is empty, draw from stock
    if not game_state.discard_pile:
        return {"play": "draw stock"}
        
    # Check if we should take the top discard card
    if game_state.should_take_discard():
        return {"play": "draw discard"}
    
    return {"play": "draw stock"}

@app.post("/lay-down/")
async def lay_down(update_info: UpdateInfo):
    """Handle playing melds and discarding with the following strategy:
    1. Meld Selection:
       - Find all possible sets and runs
       - Score each meld based on point value
       - Prioritize melds that leave good cards in hand
    
    2. Layoff Opportunities:
       - Check each card against existing melds
       - Prioritize laying off high-value cards
       - Consider both set and run layoffs
    
    3. Discard Strategy:
       - Evaluate each card's potential for future melds
       - Consider opponent's visible melds
       - Prefer discarding high-value cards not part of potential melds
    """
    global game_state
    process_events(update_info.event)
    
    if not game_state.hand:
        return {"status": "OK"}
    
    # Get all possible plays
    best_melds = game_state.find_best_meld_combination()
    
    # Check if we can lay off on existing melds
    layoffs = game_state.find_layoff_opportunities()
    
    # Construct the play string
    play_parts = []
    
    # Add melds if we have any
    for meld, score in best_melds:
        if len(meld) >= 3:  # Valid meld must have at least 3 cards
            # Validate all cards in meld
            all_valid = True
            for card in meld:
                if not game_state.validate_play(card, 'meld'):
                    all_valid = False
                    break
            
            if all_valid:
                play_parts.append("meld " + " ".join(meld))
                for card in meld:
                    if card in game_state.hand:
                        game_state.hand.remove(card)
    
    # Add layoffs
    for card, meld_num in layoffs:
        # Validate layoff is legal
        if game_state.validate_play(card, 'layoff', meld_num):
            play_parts.append(f"layoff meld({meld_num}) {card}")
            game_state.hand.remove(card)
    
    # Choose a card to discard
    discard_card = game_state.find_best_discard()
    if discard_card and game_state.validate_play(discard_card, 'discard'):
        play_parts.append(f"discard {discard_card}")
        if discard_card in game_state.hand:
            game_state.hand.remove(discard_card)
    
    if not play_parts:
        # If we can't do anything else, just discard
        if game_state.hand:
            discard_card = game_state.find_best_discard()
            play_parts.append(f"discard {discard_card}")
            if discard_card in game_state.hand:
                game_state.hand.remove(discard_card)
    
    play_string = " ".join(play_parts)
    logging.info(f"Playing: {play_string}")
    return {"play": play_string}

@app.get("/shutdown")
async def shutdown_API():
    ''' Game Server calls this endpoint to shut down the player's client after testing is completed.  Only used if DEBUG is True. '''
    os.kill(os.getpid(), signal.SIGTERM)
    logging.info("Player client shutting down...")
    return fastapi.Response(status_code=200, content='Server shutting down...')


''' Main code here - registers the player with the server via API call, and then launches the API to receive game information '''
if __name__ == "__main__":

    if (DEBUG):
        url = "http://127.0.0.1:16200/test"

        # TODO - Change logging.basicConfig if you want
        logging.basicConfig(level=logging.INFO)
    else:
        url = "http://127.0.0.1:16200/register"
        # TODO - Change logging.basicConfig if you want
        logging.basicConfig(level=logging.WARNING)

    payload = {
        "name": USER_NAME,
        "address": "127.0.0.1",
        "port": str(PORT)
    }

    try:
        # Call the URL to register client with the game server
        response = requests.post(url, json=payload)
    except Exception as e:
        print("Failed to connect to server.  Please contact Mr. Dole.")
        exit(1)

    if response.status_code == 200:
        print("Request succeeded.")
        print("Response:", response.json())  # or response.text
    else:
        print("Request failed with status:", response.status_code)
        print("Response:", response.text)
        exit(1)

    # run the client API using uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT)
