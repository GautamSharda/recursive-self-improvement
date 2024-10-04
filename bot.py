import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import product
import math
import json
import time
from collections import defaultdict

class DualNetwork(nn.Module):
    def __init__(self, input_channels, board_size, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * board_size * board_size, 256)
        self.fc_policy = nn.Linear(256, action_size)
        self.fc_value = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        policy = self.fc_policy(x)
        value = self.fc_value(x)
        return F.log_softmax(policy, dim=1), torch.tanh(value)

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0

class MCTS:
    def __init__(self, model, num_simulations, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        root = Node(root_state)
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node.children:
                action, node = self.select_child(node)
                search_path.append(node)
            
            parent = search_path[-2]
            state = parent.state
            action = node.action
            next_state = self.get_next_state(state, action)
            
            policy, value = self.evaluate(next_state)
            self.expand(node, next_state, policy)
            
            self.backpropagate(search_path, value, next_state)
        
        return self.select_action(root)

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb_score = self.ucb_score(node, child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def ucb_score(self, parent, child):
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value_sum / child.visit_count if child.visit_count > 0 else 0
        return prior_score + value_score

    def evaluate(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.model(state_tensor)
        return policy.exp().squeeze().detach().numpy(), value.item()

    def expand(self, node, state, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.get_next_state(state, action)
                child = Node(child_state, action=action, parent=node)
                child.prior = prob
                node.children[action] = child

    def backpropagate(self, search_path, value, state):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value  # Flip the value for the opponent

    def select_action(self, root):
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        actions = [action for action in root.children.keys()]
        action = actions[np.argmax(visit_counts)]
        return action

    def get_next_state(self, state, action):
        new_state = state.copy()
        y = action // (self.board_size * self.num_colors)
        x = (action % (self.board_size * self.num_colors)) // self.num_colors
        color = action % self.num_colors
        new_state[y, x] = color
        return new_state

class ARCSolver:
    def __init__(self, board_size=30, num_colors=10, num_simulations=800):
        self.board_size = board_size
        self.num_colors = num_colors
        self.model = DualNetwork(1, board_size, board_size * board_size * num_colors)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.mcts = MCTS(self.model, num_simulations)

    def train(self, puzzles, num_epochs=100, batch_size=32):
        for epoch in range(num_epochs):
            random.shuffle(puzzles)
            total_loss = 0
            for i in range(0, len(puzzles), batch_size):
                batch = puzzles[i:i+batch_size]
                loss = self.train_batch(batch)
                total_loss += loss
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(puzzles)}")

    def train_batch(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        states = torch.FloatTensor([puzzle['input'] for puzzle in batch])
        target_policies = torch.FloatTensor([self.get_target_policy(puzzle) for puzzle in batch])
        target_values = torch.FloatTensor([self.get_target_value(puzzle) for puzzle in batch])
        
        policies, values = self.model(states)
        
        policy_loss = F.kl_div(policies, target_policies)
        value_loss = F.mse_loss(values.squeeze(), target_values)
        loss = policy_loss + value_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def get_target_policy(self, puzzle):
        input_grid = puzzle['input']
        output_grid = puzzle['output']
        policy = np.zeros(self.board_size * self.board_size * self.num_colors)
        
        for y in range(min(input_grid.shape[0], output_grid.shape[0])):
            for x in range(min(input_grid.shape[1], output_grid.shape[1])):
                if input_grid[y, x] != output_grid[y, x]:
                    action = y * self.board_size * self.num_colors + x * self.num_colors + output_grid[y, x]
                    policy[action] = 1.0
        
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        else:
            policy.fill(1.0 / len(policy))
        
        return policy

    def get_target_value(self, puzzle):
        input_grid = puzzle['input']
        output_grid = puzzle['output']
        correct_pixels = np.sum(input_grid[:output_grid.shape[0], :output_grid.shape[1]] == output_grid)
        total_pixels = output_grid.size
        return correct_pixels / total_pixels

    def solve_puzzle(self, puzzle):
        self.model.eval()
        state = puzzle['input'].copy()
        output_grid = puzzle['output']
        max_steps = output_grid.size  # Maximum number of steps is the number of pixels in the output

        for _ in range(max_steps):
            action = self.mcts.search(state)
            state = self.mcts.get_next_state(state, action)
            
            # Check if the puzzle is solved
            if np.array_equal(state[:output_grid.shape[0], :output_grid.shape[1]], output_grid):
                return True  # Puzzle solved

        return False  # Puzzle not solved within the maximum number of steps

    def evaluate(self, test_puzzles):
        solved = 0
        for puzzle in test_puzzles:
            if self.solve_puzzle(puzzle):
                solved += 1
        return solved / len(test_puzzles)

def load_arc_puzzles(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    puzzles = []
    
    # Process training examples
    for train_example in json_data['train']:
        puzzles.append({
            'input': np.array(train_example['input']),
            'output': np.array(train_example['output'])
        })
    
    # Process test example
    test_example = json_data['test'][0]
    puzzles.append({
        'input': np.array(test_example['input']),
        'output': np.array(test_example['output'])
    })
    
    return puzzles

def calculate_pixel_accuracy(predicted, actual):
    return np.mean(predicted == actual) * 100

def evaluate_puzzle(solver, puzzle):
    state = solver.initialize_state(puzzle)
    for _ in range(100):  # Limit to 100 steps
        action = solver.mcts.search(state)
        if action is None:
            print("No valid action found. Stopping puzzle evaluation.")
            break
        state = solver.apply_action(state, action)
    return state[:puzzle['output'].shape[0], :puzzle['output'].shape[1]]

if __name__ == "__main__":
    # Load puzzles
    all_puzzles = load_puzzles()
    
    # Shuffle the puzzles
    random.shuffle(all_puzzles)
    
    # Split into train and test sets
    train_size = int(0.8 * len(all_puzzles))  # Use 80% for training
    train_puzzles = all_puzzles[:train_size]
    test_puzzles = all_puzzles[train_size:]

    # Initialize the solver
    solver = ARCSolver()

    # Train the solver
    solver.train(train_puzzles, num_epochs=100, batch_size=32)

    # Evaluate the solver on test puzzles
    accuracy = solver.evaluate(test_puzzles)
    print(f"Test accuracy: {accuracy}")