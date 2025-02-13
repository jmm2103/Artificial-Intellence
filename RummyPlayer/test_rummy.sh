#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Test 1: Check if server is running${NC}"
curl http://127.0.0.1:10800/

echo -e "\n${BLUE}Test 2: Start new game with a potential set${NC}"
curl -X POST http://127.0.0.1:10800/start-2p-game/ \
-H "Content-Type: application/json" \
-d '{"game_id": "test1", "opponent": "TestPlayer", "hand": "2C 2H 2S 3C 4C 5C 6H 7H 8H 9H"}'

sleep 2

echo -e "\n${BLUE}Test 3: Test drawing when there's a good card on discard${NC}"
curl -X POST http://127.0.0.1:10800/draw/ \
-H "Content-Type: application/json" \
-d '{"game_id": "test1", "event": "TestPlayer discards 2D"}'

sleep 2

echo -e "\n${BLUE}Test 4: Test laying down a set${NC}"
curl -X POST http://127.0.0.1:10800/lay-down/ \
-H "Content-Type: application/json" \
-d '{"game_id": "test1", "event": "Your turn to lay down"}'

sleep 2

echo -e "\n${BLUE}Test 5: Start new hand with a potential run${NC}"
curl -X POST http://127.0.0.1:10800/start-2p-hand/ \
-H "Content-Type: application/json" \
-d '{"hand": "3H 4H 5H 6H 7C 8C 9C TD JD QD"}'

sleep 2

echo -e "\n${BLUE}Test 6: Test drawing with no good discard option${NC}"
curl -X POST http://127.0.0.1:10800/draw/ \
-H "Content-Type: application/json" \
-d '{"game_id": "test1", "event": "TestPlayer discards KS"}'

sleep 2

echo -e "\n${BLUE}Test 7: Test laying down a run${NC}"
curl -X POST http://127.0.0.1:10800/lay-down/ \
-H "Content-Type: application/json" \
-d '{"game_id": "test1", "event": "Your turn to lay down"}'

echo -e "\n${GREEN}All tests completed!${NC}"
