def check_strike(board, player_id, coins_strike_quantity):
    """
    Check board for strike of coins_strike_quantity length.
    """
    NUM_ROWS = len(board)
    NUM_COLUMNS = len(board[0])
    
    # rows (horizontal)
    for row_index in range(NUM_ROWS):
        for col_index_start in range(0, NUM_COLUMNS - coins_strike_quantity + 1):
            strike = 0
            for col_index in range(col_index_start, col_index_start + coins_strike_quantity):
                if board[row_index][col_index] == player_id:
                    strike += 1
                    if strike == coins_strike_quantity:
                        return player_id

    # columns (vertical)
    for col_index in range(NUM_COLUMNS):
        for row_index_start in range(0, NUM_ROWS - coins_strike_quantity + 1):
            strike = 0
            for row_index in range(row_index_start, row_index_start + coins_strike_quantity):
                if board[row_index][col_index] == player_id:
                    strike += 1
                    if strike == coins_strike_quantity:
                        return player_id
                else:
                    break
                    
    # diagonal
    for row_index_start in range(0, NUM_ROWS - coins_strike_quantity + 1):
        for col_index_start in range(0, NUM_COLUMNS - coins_strike_quantity + 1):
            # Check diagonal starting from board[row_index][col_index]
            # Negative diagonal \
            strike = 0
            for i in range(coins_strike_quantity):
                if board[row_index_start + i][col_index_start + i] == player_id:
                    strike += 1
                    if strike == coins_strike_quantity:
                        return player_id
                else:
                    break
                
            # Positive diagonal /
            strike = 0
            for i in range(coins_strike_quantity):
                if board[row_index_start + i][NUM_COLUMNS - col_index_start - i - 1] == player_id:
                    strike += 1
                    if strike == coins_strike_quantity:
                        return player_id
                else:
                    break
    return False