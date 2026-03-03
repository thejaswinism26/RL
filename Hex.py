import numpy as np

class Hex:
    def __init__(self, size=5):
        self.size = size
        self.row_count = size
        self.column_count = size
        self.action_size = size * size
        
    def __repr__(self):
        return f"Hex_{self.size}"
        
    def get_initial_state(self):
        return np.zeros((self.size, self.size))
    
    def get_next_state(self, state, action, player):
        row = action // self.size
        col = action % self.size
        state[row, col] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        return encoded_state
    
    def get_neighbors(self, row, col):
        # Hex grid neighbors (6 directions)
        directions = [
            (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0)
        ]
        
        neighbors = []
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size:
                neighbors.append((r, c))
        return neighbors
    
    def check_win(self, state, player):
        visited = set()
        stack = []
        
        if player == 1:
            # connect top to bottom
            for col in range(self.size):
                if state[0, col] == player:
                    stack.append((0, col))
                    visited.add((0, col))
                    
            target_row = self.size - 1
            
            while stack:
                r, c = stack.pop()
                if r == target_row:
                    return True
                
                for nr, nc in self.get_neighbors(r, c):
                    if state[nr, nc] == player and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
                        
        else:
            # connect left to right
            for row in range(self.size):
                if state[row, 0] == player:
                    stack.append((row, 0))
                    visited.add((row, 0))
                    
            target_col = self.size - 1
            
            while stack:
                r, c = stack.pop()
                if c == target_col:
                    return True
                
                for nr, nc in self.get_neighbors(r, c):
                    if state[nr, nc] == player and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
                        
        return False
    
    def get_value_and_terminated(self, state, action):
        if action is None:
            return 0, False
        
        row = action // self.size
        col = action % self.size
        player = state[row, col]
        
        if self.check_win(state, player):
            return 1, True
        
        return 0, False  # Hex cannot draw-