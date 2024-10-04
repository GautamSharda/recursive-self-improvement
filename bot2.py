import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, defaultdict
import json
import os
import random
import logging
import sys

# Set up logging
def setup_logging():
    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    summary_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create handlers
    detailed_handler = logging.FileHandler('detailed_log.txt', mode='w')
    summary_handler = logging.FileHandler('summary_log.txt', mode='w')
    console_handler = logging.StreamHandler(sys.stdout)

    # Set formatters for handlers
    detailed_handler.setFormatter(detailed_formatter)
    summary_handler.setFormatter(summary_formatter)
    console_handler.setFormatter(detailed_formatter)

    # Create loggers
    detailed_logger = logging.getLogger('detailed')
    summary_logger = logging.getLogger('summary')

    # Set levels
    detailed_logger.setLevel(logging.DEBUG)
    summary_logger.setLevel(logging.INFO)

    # Add handlers to loggers
    detailed_logger.addHandler(detailed_handler)
    detailed_logger.addHandler(console_handler)
    summary_logger.addHandler(summary_handler)

    return detailed_logger, summary_logger

detailed_logger, summary_logger = setup_logging()

def log_grid(grid, grid_name):
    grid_str = '\n'.join([' '.join(map(str, row)) for row in grid])
    detailed_logger.debug(f"{grid_name}:\n{grid_str}")

# Define the grid environment for ARC tasks
class ARCEnvironment:
    def __init__(self, input_grid, target_grid):
        self.input_grid = input_grid
        self.target_grid = target_grid
        self.current_grid = np.copy(input_grid)
        self.height, self.width = input_grid.shape
        self.colors = set(np.unique(input_grid)) | set(np.unique(target_grid))
        self.action_size = self.height * self.width * len(self.colors)

    def reset(self):
        self.current_grid = np.copy(self.input_grid)
        return self.current_grid

    def step(self, action):
        x, y, c = self.action_to_tuple(action)
        self.current_grid[x, y] = c
        reward = -np.sum(self.current_grid != self.target_grid)
        done = np.array_equal(self.current_grid, self.target_grid)
        return self.current_grid, reward, done

    def render(self):
        print("Current Grid State:")
        print(self.current_grid)
        print("Target Grid:")
        print(self.target_grid)

    def action_to_tuple(self, action):
        x = action // (self.width * len(self.colors))
        y = (action % (self.width * len(self.colors))) // len(self.colors)
        c = list(self.colors)[action % len(self.colors)]
        return (x, y, c)

class FlexibleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FlexibleConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv(x)
        x = self.adaptive_pool(x)
        return x

class MuZeroARCNet(nn.Module):
    def __init__(self, input_channels, hidden_size):
        super(MuZeroARCNet, self).__init__()
        self.hidden_size = hidden_size
        self.action_size = None  # Will be set later

        self.representation = nn.Sequential(
            FlexibleConv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            FlexibleConv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            FlexibleConv2d(64, hidden_size, kernel_size=3, padding=1),
            nn.Flatten(),
            nn.Linear(hidden_size, hidden_size)  # Ensure output is always hidden_size
        )

        self.dynamics = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Policy and value heads will be created in update_action_size method
        self.policy = None
        self.value = None

    def update_action_size(self, new_action_size):
        self.action_size = new_action_size
        self.policy = nn.Linear(self.hidden_size, self.action_size)
        self.value = nn.Linear(self.hidden_size, 1)

    def initial_inference(self, state):
        hidden = self.representation(state)
        policy_logits = self.policy(hidden)
        value = self.value(hidden)
        return hidden, policy_logits, value

    def recurrent_inference(self, hidden_state, action):
        action_encoded = torch.zeros(1, 1)  # Changed from self.action_size to 1
        action_encoded[0, 0] = action
        x = torch.cat([hidden_state, action_encoded], dim=1)
        next_hidden = self.dynamics(x)
        policy_logits = self.policy(next_hidden)
        value = self.value(next_hidden)
        return next_hidden, policy_logits, value

# Define MCTS node for ARC
class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None  # State represented by this node

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, policy):
        for action, p in enumerate(policy):
            if p > 0:
                self.children[action] = Node(p)

    def select_child(self):
        c_puct = 1.0
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = child.value() + c_puct * child.prior * (self.visit_count ** 0.5) / (child.visit_count + 1)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

class MCTS:
    def __init__(self, network):
        self.network = network
        self.action_size = None  # Will be set later

    def run(self, state, num_simulations):
        root = Node(0)
        hidden, policy_logits, value = self.network.initial_inference(state)
        policy = torch.softmax(policy_logits, dim=1)
        root.expand(range(self.action_size), policy[0].detach().numpy())

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            current_hidden = hidden

            # Selection
            while node.expanded():
                action, node = node.select_child()
                search_path.append(node)
                current_hidden, policy_logits, value = self.network.recurrent_inference(current_hidden, action)

            # Expansion
            policy = torch.softmax(policy_logits, dim=1)
            node.expand(range(self.action_size), policy[0].detach().numpy())

            # Backpropagation
            self.backpropagate(search_path, value.item())

        return root

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1

def load_arc_tasks(directory):
    tasks = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                task = json.load(f)
                task['filename'] = filename  # Store filename for printing
                tasks.append(task)
    print(f"Loaded {len(tasks)} ARC tasks from {directory}")
    return tasks

def get_random_arc_task(tasks):
    task = random.choice(tasks)
    train_pair = random.choice(task['train'])
    input_grid = np.array(train_pair['input'])
    output_grid = np.array(train_pair['output'])
    return task['filename'], input_grid, output_grid

def calculate_pixel_correctness(output_grid, target_grid):
    if output_grid.shape != target_grid.shape:
        detailed_logger.error(f"Shape mismatch: output_grid {output_grid.shape}, target_grid {target_grid.shape}")
        return 0.0

    total_pixels = output_grid.size
    correct_pixels = np.sum(output_grid == target_grid)
    correctness = (correct_pixels / total_pixels) * 100

    if correctness > 100:
        detailed_logger.error(f"Correctness > 100%: {correctness}%")
        detailed_logger.error(f"Output grid:\n{output_grid}")
        detailed_logger.error(f"Target grid:\n{target_grid}")
        return 100.0

    return correctness

# Training loop for MuZero on ARC tasks
def train_muzero_arc(num_iterations=1000, num_simulations=50, arc_tasks=None, save_interval=100):
    hidden_size = 64
    network = MuZeroARCNet(input_channels=1, hidden_size=hidden_size)
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    mcts = MCTS(network)

    total_reward = 0
    solved_tasks = 0
    total_pixel_correctness = 0

    # Create a directory for saving model weights
    os.makedirs('model_weights', exist_ok=True)

    for iteration in range(num_iterations):
        if arc_tasks:
            filename, input_grid, target_grid = get_random_arc_task(arc_tasks)
            detailed_logger.info(f"Iteration {iteration + 1}/{num_iterations}: Training on task {filename}")
            summary_logger.info(f"Iteration {iteration + 1}/{num_iterations}: Training on task {filename}")
        else:
            height, width = np.random.randint(3, 10, size=2)
            input_grid = np.random.randint(0, 10, (height, width))
            target_grid = np.random.randint(0, 10, (height, width))
            detailed_logger.info(f"Iteration {iteration + 1}/{num_iterations}: Training on random grid")
            summary_logger.info(f"Iteration {iteration + 1}/{num_iterations}: Training on random grid")
        
        env = ARCEnvironment(input_grid, target_grid)
        
        log_grid(input_grid, "Input grid")
        log_grid(target_grid, "Target grid")
        
        action_size = env.action_size
        network.update_action_size(action_size)
        mcts.action_size = action_size

        state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Run MCTS
        root = mcts.run(state, num_simulations)

        # Select action based on MCTS policy
        action_probs = [child.visit_count / root.visit_count for child in root.children.values()]
        action = np.random.choice(action_size, p=action_probs)

        # Take a step in the environment
        next_state, reward, done = env.step(action)

        log_grid(env.current_grid, "Current output grid")
        log_grid(target_grid, "Target grid")  # Log target grid for comparison

        pixel_correctness = calculate_pixel_correctness(env.current_grid, target_grid)
        total_pixel_correctness += pixel_correctness
        detailed_logger.info(f"Iteration {iteration + 1}: Pixel correctness: {pixel_correctness:.2f}%")
        summary_logger.info(f"Iteration {iteration + 1}: Pixel correctness: {pixel_correctness:.2f}%")

        total_reward += reward
        if done:
            solved_tasks += 1
            detailed_logger.info(f"Task solved! Total solved: {solved_tasks}")
            summary_logger.info(f"Task solved! Total solved: {solved_tasks}")

        # Train the network
        target_value = reward
        target_policy = torch.tensor(action_probs, dtype=torch.float32)
        
        hidden, policy_logits, value = network.initial_inference(state)
        loss = nn.MSELoss()(value, torch.tensor([[target_value]], dtype=torch.float32)) + \
               nn.CrossEntropyLoss()(policy_logits, target_policy.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iteration + 1) % 10 == 0:  # Log every 10 iterations
            avg_pixel_correctness = total_pixel_correctness / (iteration + 1)
            log_message = (f"Iteration {iteration + 1}: Loss = {loss.item():.4f}, "
                           f"Average Reward = {total_reward / (iteration + 1):.4f}, "
                           f"Solved Tasks = {solved_tasks}, "
                           f"Average Pixel Correctness = {avg_pixel_correctness:.2f}%")
            detailed_logger.info(log_message)
            summary_logger.info(log_message)

        if (iteration + 1) % save_interval == 0:
            # Save model weights periodically
            torch.save(network.state_dict(), f'model_weights/muzero_arc_model_iter_{iteration+1}.pth')
            detailed_logger.info(f"Model weights saved at iteration {iteration+1}")
            summary_logger.info(f"Model weights saved at iteration {iteration+1}")

    # Save final model weights
    torch.save(network.state_dict(), 'model_weights/muzero_arc_model_final.pth')
    detailed_logger.info("Final model weights saved")
    summary_logger.info("Final model weights saved")

    avg_pixel_correctness = total_pixel_correctness / num_iterations
    final_message = (f"Training completed. Total solved tasks: {solved_tasks}/{num_iterations}\n"
                     f"Average reward per iteration: {total_reward / num_iterations:.4f}\n"
                     f"Average pixel correctness: {avg_pixel_correctness:.2f}%")
    detailed_logger.info(final_message)
    summary_logger.info(final_message)

    return network

def test_muzero_arc(arc_tasks=None):
    print("Starting MuZero ARC test...")
    
    num_iterations = 5
    num_simulations = 10
    
    try:
        trained_network = train_muzero_arc(num_iterations, num_simulations, arc_tasks=arc_tasks)
        print("Test completed successfully!")
        return True
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    arc_directory = "data/training"  # Replace with the actual path
    arc_tasks = load_arc_tasks(arc_directory)

    detailed_logger.info("Starting MuZero ARC test...")
    summary_logger.info("Starting MuZero ARC test...")

    test_successful = test_muzero_arc(arc_tasks=arc_tasks)
    if test_successful:
        detailed_logger.info("Test passed. Starting full training run...")
        summary_logger.info("Test passed. Starting full training run...")
        full_num_iterations = 1000  # Adjust as needed
        full_num_simulations = 50   # Adjust as needed
        
        try:
            full_trained_network = train_muzero_arc(full_num_iterations, full_num_simulations, arc_tasks=arc_tasks)
            detailed_logger.info("Full training completed successfully!")
            summary_logger.info("Full training completed successfully!")

            # Save the final model
            torch.save(full_trained_network.state_dict(), 'model_weights/muzero_arc_model_final.pth')
            detailed_logger.info("Final model saved successfully!")
            summary_logger.info("Final model saved successfully!")

        except Exception as e:
            detailed_logger.error(f"Full training run failed with error: {str(e)}")
            summary_logger.error(f"Full training run failed with error: {str(e)}")
    else:
        detailed_logger.warning("Test failed. Please fix any issues before running a full training session.")
        summary_logger.warning("Test failed. Please fix any issues before running a full training session.")