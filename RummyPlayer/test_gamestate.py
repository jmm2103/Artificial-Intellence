from GameState import GameState

def test_can_lay_off_on_meld():
    # Create a game state instance
    gs = GameState()
    
    # Test 1: Laying off on a set
    print("\nTest 1: Laying off on a set")
    meld = ["2H", "2C", "2D"]
    card = "2S"
    result = gs.can_lay_off_on_meld(card, meld)
    print(f"Can lay {card} on set {meld}? {result}")
    assert result == True, "Should be able to lay off matching rank"
    
    # Test 2: Invalid layoff on a set
    print("\nTest 2: Invalid layoff on a set")
    card = "3H"
    result = gs.can_lay_off_on_meld(card, meld)
    print(f"Can lay {card} on set {meld}? {result}")
    assert result == False, "Should not be able to lay off different rank"
    
    # Test 3: Laying off on a run (at end)
    print("\nTest 3: Laying off on a run (at end)")
    meld = ["2H", "3H", "4H"]
    card = "5H"
    result = gs.can_lay_off_on_meld(card, meld)
    print(f"Can lay {card} on run {meld}? {result}")
    assert result == True, "Should be able to lay off at end of run"
    
    # Test 4: Laying off on a run (at start)
    print("\nTest 4: Laying off on a run (at start)")
    card = "AH"
    result = gs.can_lay_off_on_meld(card, meld)
    print(f"Can lay {card} on run {meld}? {result}")
    assert result == False, "Should not be able to lay off A on 2"
    
    # Test 5: Invalid layoff on a run (wrong suit)
    print("\nTest 5: Invalid layoff on a run (wrong suit)")
    card = "5S"
    result = gs.can_lay_off_on_meld(card, meld)
    print(f"Can lay {card} on run {meld}? {result}")
    assert result == False, "Should not be able to lay off different suit"
    
    # Test 6: Empty meld
    print("\nTest 6: Empty meld")
    result = gs.can_lay_off_on_meld(card, [])
    print(f"Can lay {card} on empty meld? {result}")
    assert result == False, "Should not be able to lay off on empty meld"
    
    print("\nAll tests passed!")

# Comment out all other test functions
def test_calculate_hand_value():
    # Create a game state instance
    gs = GameState()
    
    # Test 1: Mix of number and face cards
    print("\nTest 1: Mix of number and face cards")
    test_hand = ["2C", "TH", "KS", "AD", "5H"]
    gs.set_hand(test_hand)
    value = gs.calculate_hand_value()
    print(f"Hand {test_hand} value: {value}")
    assert value == 37, f"Expected 37, got {value}"
    
    # Test 2: All face cards
    print("\nTest 2: All face cards")
    test_hand = ["TH", "JS", "QC", "KD", "AC"]
    gs.set_hand(test_hand)
    value = gs.calculate_hand_value()
    print(f"Hand {test_hand} value: {value}")
    assert value == 50, f"Expected 50, got {value}"
    
    # Test 3: All number cards
    print("\nTest 3: All number cards")
    test_hand = ["2C", "3H", "4S", "5D", "6C"]
    gs.set_hand(test_hand)
    value = gs.calculate_hand_value()
    print(f"Hand {test_hand} value: {value}")
    assert value == 20, f"Expected 20, got {value}"
    
    print("All calculate_hand_value tests passed!")

def test_find_best_meld_combination():
    # Create a game state instance
    gs = GameState()
    
    # Test 1: Hand with a set
    print("\nTest 1: Hand with a set")
    test_hand = ["2C", "2H", "2S", "KD", "5H"]
    gs.set_hand(test_hand)
    melds = gs.find_best_meld_combination()
    print(f"Hand {test_hand}\nMelds found: {melds}")
    assert len(melds) == 1, "Should find one meld"
    assert len(melds[0][0]) == 3, "Meld should have 3 cards"
    
    # Test 2: Hand with a run
    print("\nTest 2: Hand with a run")
    test_hand = ["2H", "3H", "4H", "KD", "5S"]
    gs.set_hand(test_hand)
    melds = gs.find_best_meld_combination()
    print(f"Hand {test_hand}\nMelds found: {melds}")
    assert len(melds) == 1, "Should find one meld"
    assert len(melds[0][0]) == 3, "Meld should have 3 cards"
    
    print("All find_best_meld_combination tests passed!")

def test_find_best_discard():
    # Create a game state instance
    gs = GameState()
    
    # Test 1: Hand with obvious discard
    print("\nTest 1: Hand with obvious discard")
    test_hand = ["2C", "2H", "2S", "KD", "5H"]
    gs.set_hand(test_hand)
    discard = gs.find_best_discard()
    print(f"Hand {test_hand}\nBest discard: {discard}")
    assert discard in ["KD", "5H"], "Should discard high card not in meld"
    
    # Test 2: Hand with potential run
    print("\nTest 2: Hand with potential run")
    test_hand = ["2H", "3H", "4H", "KD", "6H"]
    gs.set_hand(test_hand)
    discard = gs.find_best_discard()
    print(f"Hand {test_hand}\nBest discard: {discard}")
    assert discard == "KD", "Should discard KD and keep potential run cards"
    
    # Test 3: Hand with high cards but potential meld
    print("\nTest 3: Hand with high cards but potential meld")
    test_hand = ["TH", "TC", "KD", "KH", "3S"]
    gs.set_hand(test_hand)
    discard = gs.find_best_discard()
    print(f"Hand {test_hand}\nBest discard: {discard}")
    assert discard == "3S", "Should keep pairs of high cards over low single card"
    
    print("All find_best_discard tests passed!")

def test_find_layoff_opportunities():
    # Create a game state instance
    gs = GameState()
    
    # Test 1: Layoff on a set
    print("\nTest 1: Layoff on a set")
    gs.hand = ["2S", "3H"]
    gs.meld_piles = {1: ["2H", "2C", "2D"]}
    layoffs = gs.find_layoff_opportunities()
    print(f"Hand {gs.hand}\nMeld piles: {gs.meld_piles}\nLayoff opportunities: {layoffs}")
    assert len(layoffs) == 1, "Should find one layoff opportunity"
    assert layoffs[0][0] == "2S", "Should identify 2S as layoff card"
    
    # Test 2: Layoff on a run
    print("\nTest 2: Layoff on a run")
    gs.hand = ["5H", "AH"]
    gs.meld_piles = {1: ["2H", "3H", "4H"]}
    layoffs = gs.find_layoff_opportunities()
    print(f"Hand {gs.hand}\nMeld piles: {gs.meld_piles}\nLayoff opportunities: {layoffs}")
    assert len(layoffs) == 1, "Should find one layoff opportunity"
    assert layoffs[0][0] == "5H", "Should identify 5H as layoff card"
    
    print("All find_layoff_opportunities tests passed!")


if __name__ == "__main__":
    print("Running all tests...\n")
    test_calculate_hand_value()
    test_find_best_meld_combination()
    test_find_best_discard()
    test_find_layoff_opportunities()
    print("\nAll tests completed successfully!")
