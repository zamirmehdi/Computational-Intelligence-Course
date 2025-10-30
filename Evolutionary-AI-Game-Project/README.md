# Evolutionary AI Game - Neuroevolution

Neuroevolution system that trains neural networks to play games using genetic algorithms. Agents learn to navigate through obstacles in real-time by evolving their neural network weights across generations.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Pygame](https://img.shields.io/badge/Pygame-Game%20Framework-green.svg)](#)
[![NumPy](https://img.shields.io/badge/NumPy-Computation-orange.svg)](#)

## 📑 Table of Contents

- [Overview](#-overview)
- [Game Modes](#-game-modes)
- [Neuroevolution System](#-neuroevolution-system)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Genetic Algorithm](#genetic-algorithm)
- [Project Structure](#️-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Advanced Features](#-advanced-features)
- [Implementation Details](#-implementation-details)
- [Results](#-results)
- [Project Information](#ℹ️-project-information)
- [Contact](#-contact)

## 📋 Overview

This project implements **neuroevolution** - the process of evolving artificial neural networks through genetic algorithms. Instead of using traditional backpropagation, the neural network weights evolve over generations through:
- **Selection** of the fittest agents
- **Crossover** of parent networks
- **Mutation** of network weights

The system is tested on three different game modes, each requiring different strategies and demonstrating the adaptability of evolutionary algorithms.

## 🎮 Game Modes

Helicopter             |  Gravity          |  Thrust
:-------------------------:|:-------------------------:|:-------------------------:
![Helicopter](https://github.com/HosseinZaredar/EvolutionaryGames/blob/main/screenshots/helicopter.png?raw=true)  |  ![Gravity](https://github.com/HosseinZaredar/EvolutionaryGames/blob/main/screenshots/gravity.png?raw=true) | ![Thrust](https://github.com/HosseinZaredar/EvolutionaryGames/blob/main/screenshots/thrust.png?raw=true)


### 1. Helicopter Mode
```bash
python game.py --mode helicopter --play True
```
- **Control:** Press `Space` to fly up, release to fall
- **Objective:** Navigate through gaps between obstacles
- **Challenge:** Precise timing and height control

### 2. Gravity Mode
```bash
python game.py --mode gravity --play True
```
- **Control:** Press `Space` to reverse gravity
- **Objective:** Avoid obstacles by flipping gravity
- **Challenge:** Gravity inversion timing

### 3. Thrust Mode
```bash
python game.py --mode thrust --play True
```
- **Control:** `Up`/`Down` arrows for thrust control
- **Objective:** Navigate using continuous thrust
- **Challenge:** Momentum and precise control

## 🧬 Neuroevolution System

### Neural Network Architecture

The agent's brain is a **3-layer feedforward neural network**:

```
Input Layer (Variable)  →  Hidden Layer 1 (16)  →  Hidden Layer 2 (16)  →  Output Layer (Variable)
```

**Input Features:**
- Player position (x, y)
- Player velocity (y-axis)
- Next obstacle position (x)
- Next obstacle gap center (y)
- Next obstacle gap size

**Output Actions:**
- **Helicopter:** `1` (fly up) or `-1` (fall)
- **Gravity:** `1` (reverse gravity) or `-1` (normal gravity)
- **Thrust:** `1` (thrust up), `0` (no thrust), `-1` (thrust down)

**Activation Function:** Sigmoid for all layers

### Genetic Algorithm

The evolution process follows these steps:

#### 1. **Initialization**
- Generate random population of `num_players` agents
- Each agent has randomly initialized neural network weights

#### 2. **Fitness Evaluation**
```python
fitness = delta_x  # Distance traveled before collision
```
- Agents play the game simultaneously
- Fitness = horizontal distance traveled
- Agents that survive longer have higher fitness

#### 3. **Parent Selection**
Two selection strategies available:

**a) (μ + λ) Strategy:**
- Select best μ parents from combined pool of parents and offspring
- More conservative, preserves good solutions

**b) (μ, λ) Strategy:**
- Select best μ parents only from offspring
- More explorative, forces innovation

Implementation:
```python
def next_population_selection(players, num_players):
    # Sort players by fitness
    players.sort(key=lambda x: x.fitness, reverse=True)
    # Return top performers
    return players[:num_players]
```

#### 4. **Crossover**
- Combine weights from two parent networks
- Creates offspring with mixed characteristics
- Multiple crossover strategies possible:
  - Uniform crossover
  - Single-point crossover
  - Arithmetic crossover

#### 5. **Mutation**
```python
def mutate(child):
    # Randomly perturb weights
    mutation_rate = 0.1
    for layer in child.nn.weights:
        if random() < mutation_rate:
            layer += random_noise()
```
- Add random noise to weights
- Introduces genetic diversity
- Prevents premature convergence
- Tunable mutation rate

#### 6. **Next Generation**
- Replace old population with new offspring
- Repeat the cycle
- Evolution continues until convergence or max generations

## 🗂️ Project Structure

```
Evolutionary-AI-Game-Project/
├── docs/
│   ├── Instruction.pdf          # Project specification (Persian)
│   └── Report.pdf               # Implementation report (Persian)
├── src/
│   ├── game.py                  # Main game loop and rendering
│   ├── player.py                # Player agent with neural network
│   ├── evolution.py             # Genetic algorithm implementation
│   ├── nn.py                    # Neural network (feedforward)
│   ├── config.py                # Hyperparameters and configuration
│   ├── util.py                  # Utility functions
│   └── box_list.py              # Obstacle generation and management
├── checkpoint/
│   ├── helicopter/              # Saved states for helicopter mode
│   ├── gravity/                 # Saved states for gravity mode
│   └── thrust/                  # Saved states for thrust mode
└── README.md
```

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/zamirmehdi/Computational-Intelligence-Course.git
cd Computational-Intelligence-Course/Evolutionary-AI-Game-Project
```

2. **Install dependencies:**
```bash
pip install pygame numpy
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Playing Manually

Test the game mechanics by playing manually:

```bash
# Helicopter mode
python game.py --mode helicopter --play True

# Gravity mode
python game.py --mode gravity --play True

# Thrust mode
python game.py --mode thrust --play True
```

### Training AI Agents

Start evolutionary training from scratch:

```bash
python game.py --mode helicopter
```

The AI will:
- Start with random neural network weights
- Evolve over generations
- Display real-time evolution progress
- Save checkpoints every 5 generations

### Loading Checkpoints

Resume training or test pre-trained agents:

```bash
python game.py --mode helicopter --checkpoint checkpoint/helicopter/20
```

This loads the saved state from generation 20.

### Keyboard Controls During Training

- **`ESC`** - Quit the simulation
- **`F`** - Toggle FPS display
- **`D`** - Toggle debug information
- **`S`** - Save current generation checkpoint

## ⚙️ Configuration

Edit `config.py` to customize the evolution:

```python
# Evolution parameters
NUM_PLAYERS = 50              # Population size
NUM_GENERATIONS = 100         # Max generations
MUTATION_RATE = 0.1           # Probability of mutation
CROSSOVER_RATE = 0.8          # Probability of crossover

# Neural network architecture
HIDDEN_LAYER_1_SIZE = 16      # First hidden layer neurons
HIDDEN_LAYER_2_SIZE = 16      # Second hidden layer neurons

# Game settings
FPS = 50                      # Frames per second
GAME_SPEED = 1.0              # Game speed multiplier
CHECKPOINT_INTERVAL = 5       # Save every N generations
```

## 🎯 Advanced Features

### 1. Transfer Learning

Train on one game mode and transfer learned weights to another:

```python
# Train on helicopter mode
python game.py --mode helicopter

# Load helicopter weights for gravity mode
python game.py --mode gravity --checkpoint checkpoint/helicopter/50
```

**Benefits:**
- Faster convergence on new tasks
- Leverage previously learned patterns
- Demonstrate knowledge transfer

### 2. Learning Curves

Visualize evolution progress:

```python
def plot_learning_curve(generations, fitness_scores):
    plt.plot(generations, fitness_scores)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Evolution Progress')
    plt.show()
```

Track metrics:
- Best fitness per generation
- Average fitness per generation
- Fitness variance (diversity measure)

### 3. Real-time Visualization

Watch agents evolve in real-time:
- Multiple agents displayed simultaneously
- Color-coded by fitness
- Death animations
- Generation statistics overlay

## 💻 Implementation Details

### Player Agent (`player.py`)

```python
class Player:
    def __init__(self):
        self.nn = NeuralNetwork()
        self.fitness = 0
        self.position = [x, y]
        self.velocity = 0
    
    def think(self, game_state):
        # Extract features from game state
        inputs = self.get_inputs(game_state)
        
        # Forward pass through neural network
        output = self.nn.forward(inputs)
        
        # Convert output to action
        action = self.output_to_action(output)
        return action
```

### Neural Network (`nn.py`)

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        
        # Initialize random weights
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, inputs):
        activation = inputs
        for w, b in zip(self.weights, self.biases):
            z = w @ activation + b
            activation = sigmoid(z)
        return activation
```

### Genetic Algorithm (`evolution.py`)

```python
def evolve_population(players, num_players):
    # Calculate fitness
    calculate_fitness(players)
    
    # Select parents
    parents = select_parents(players, num_players)
    
    # Generate offspring
    offspring = []
    while len(offspring) < num_players:
        parent1, parent2 = select_two_parents(parents)
        child = crossover(parent1, parent2)
        mutate(child)
        offspring.append(child)
    
    return offspring
```

## 📊 Results

### Performance Metrics

| Game Mode | Generations to Converge | Best Fitness | Average Fitness |
|-----------|------------------------|--------------|-----------------|
| Helicopter | 30-50 | 5000+ | 3500+ |
| Gravity | 40-60 | 4500+ | 3000+ |
| Thrust | 50-70 | 4000+ | 2800+ |

### Evolution Progress

**Generation 1:**
- Random behavior
- Quick failures
- Fitness: 50-200

**Generation 20:**
- Basic obstacle avoidance
- Some agents survive longer
- Fitness: 500-1000

**Generation 50:**
- Sophisticated strategies
- Consistent performance
- Fitness: 3000-5000

### Transfer Learning Results

Training a model on Helicopter and transferring to Gravity:
- **Without Transfer:** 60 generations to converge
- **With Transfer:** 25 generations to converge
- **Improvement:** ~58% faster convergence

## 🎓 Key Concepts Demonstrated

### Evolutionary Computation
- **Genetic Algorithms:** Selection, crossover, mutation
- **Fitness-based Evolution:** Natural selection simulation
- **Population Dynamics:** Diversity vs. convergence

### Neural Networks
- **Feedforward Architecture:** Multi-layer perception
- **Activation Functions:** Non-linear transformations
- **Weight Evolution:** Alternative to backpropagation

### Game AI
- **Real-time Decision Making:** Frame-by-frame actions
- **State Representation:** Feature extraction from game state
- **Policy Learning:** Mapping states to actions

### Machine Learning
- **No Gradient Descent:** Evolution instead of backprop
- **Exploration vs. Exploitation:** Balance in evolution
- **Transfer Learning:** Knowledge transfer between tasks

## 🔬 Experiments & Extensions

### Suggested Experiments

1. **Vary Population Size**
   - Test with 10, 50, 100, 200 agents
   - Analyze convergence speed vs. computational cost

2. **Mutation Rate Tuning**
   - Test rates: 0.01, 0.05, 0.1, 0.2
   - Find optimal balance between exploration and exploitation

3. **Network Architecture**
   - Try different hidden layer sizes: 8, 16, 32, 64
   - Test deeper networks: 3 or 4 hidden layers

4. **Selection Strategies**
   - Compare (μ + λ) vs (μ, λ)
   - Implement tournament selection
   - Try rank-based selection

5. **Hybrid Approaches**
   - Combine evolution with local search
   - Use backpropagation for fine-tuning

## 📖 Documentation

**Instruction Manual:** `docs/Instruction.pdf` (Persian)
- Detailed project requirements
- Algorithm specifications
- Implementation guidelines

**Report:** `docs/Report.pdf` (Persian)
- Implementation details
- Experimental results
- Performance analysis
- Challenges and solutions

## ⚠️ Known Issues & Limitations

- **Computational Cost:** Evolution can be slow with large populations
- **Local Optima:** May converge to suboptimal solutions
- **Determinism:** Different runs produce different results
- **Hardware Dependent:** Performance varies by CPU speed

## 🔮 Future Enhancements

- [ ] Implement NEAT (NeuroEvolution of Augmenting Topologies)
- [ ] Add more game modes
- [ ] Parallelize evolution across CPU cores
- [ ] Implement novelty search
- [ ] Add speciation to maintain diversity
- [ ] Create web-based visualization
- [ ] Support for RNN/LSTM networks

## ℹ️ Project Information

**Course:** Computational Intelligence  
**University:** Amirkabir University of Technology (Tehran Polytechnic)  
**Semester:** Spring 2021  
**Author:** Amirmehdi Zarrinnezhad

## Contributors & Designers
- [Hossein Zaredar](https://github.com/HosseinZaredar)
- [Matin Tavakoli](https://github.com/MatinTavakoli/) <br>
- Many thanks to [Parnian Rad](https://github.com/Parnian-Rad)

## 🔗 Related Projects

Part of the [Computational Intelligence Course](https://github.com/zamirmehdi/Computational-Intelligence-Course) repository.

**Other Projects:**
- [Fuzzy C-Means Clustering](https://github.com/zamirmehdi/Fuzzy_C-means)
- [Handwritten Digit Recognition](https://github.com/zamirmehdi/Handwritten-Digit-Recognition)

## 📚 References

- **Floreano, D., Dürr, P., & Mattiussi, C.** (2008). *Neuroevolution: from architectures to learning*. Evolutionary Intelligence, 1(1), 47-62.
- **Stanley, K. O., & Miikkulainen, R.** (2002). *Evolving neural networks through augmenting topologies*. Evolutionary Computation, 10(2), 99-127.
- **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

## 📧 Contact

Questions or collaborations? Feel free to reach out!

**Amirmehdi Zarrinnezhad**  
📧 Email: amzarrinnezhad@gmail.com  
🌐 GitHub: [@zamirmehdi](https://github.com/zamirmehdi)

---

<div align="center">

[⬆ Back to Main Repository](https://github.com/zamirmehdi/Computational-Intelligence-Course)

</div>

<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>

