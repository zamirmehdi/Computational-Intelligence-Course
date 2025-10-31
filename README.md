<h1 align="center">Computational Intelligence</h1>#

Collection of three comprehensive projects exploring evolutionary algorithms, fuzzy logic, and neural networks from the Computational Intelligence course at Amirkabir University of Technology.

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-orange.svg)](#)
[![Pygame](https://img.shields.io/badge/Pygame-Game%20Development-green.svg)](#)

<details> <summary><h2>📚 Table of Contents</h2></summary>

- [Overview](#-overview)
- [Projects](#-projects)
  - [Evolutionary AI Game](#1-evolutionary-ai-game)
  - [Fuzzy C-Means Clustering](#2-fuzzy-c-means-clustering)
  - [Handwritten Digit Recognition](#3-handwritten-digit-recognition)
- [Repository Structure](#️-repository-structure)
- [Technologies Used](#%EF%B8%8F-technologies-used)
- [Getting Started](#-getting-started)
- [Project Information](#ℹ️-project-information)
- [Related Courses](#-related-courses)
- [Contact](#-contact)
</details>

## 📋 Overview

This repository contains three comprehensive projects completed during my undergraduate Computational Intelligence course at Amirkabir University of Technology. Each project demonstrates fundamental concepts in computational intelligence through practical, from-scratch implementations:

1. **Evolutionary AI Game** - Neuroevolution using genetic algorithms for game-playing agents
2. **Fuzzy C-Means Clustering** - Soft clustering with fuzzy logic and membership degrees
3. **Handwritten Digit Recognition** - Multi-layer neural network with backpropagation on MNIST

All projects are accompanied by their respective codes, necessary datasets (if required), and comprehensive documentation, which includes instructions and reports (in Persian).

**Course Focus:**
- Evolutionary computation and genetic algorithms
- Fuzzy logic and soft computing
- Neural networks and deep learning fundamentals
- Optimization algorithms
- Pattern recognition and classification

## 📚 Projects

### 1. Evolutionary AI Game

Neuroevolution system that trains neural networks to play games using genetic algorithms.

**Description:**
- Implements genetic algorithms to evolve neural networks for game-playing agents
- Three game modes: Helicopter, Gravity, and Thrust
- Real-time visualization of evolutionary progress
- Transfer learning capabilities across different game modes

**Key Features:**
- **Genetic Algorithm:** (μ, λ) and (μ + λ) selection strategies
- **Neural Network:** 3-layer feedforward architecture
- **Fitness Evaluation:** Distance-based scoring system
- **Evolution Operators:** Parent selection, crossover, and mutation
- **Advanced Features:** Checkpointing, learning curves, transfer learning

**Technologies:** Python, Pygame, NumPy

📂 **[View Project Details](./Evolutionary-AI-Game-Project)**

---

### 2. Fuzzy C-Means Clustering

Classification based on Fuzzy Logic (C-Means) for unsupervised data clustering.

**Description:**
- Implements Fuzzy C-Means clustering algorithm from scratch
- Soft clustering approach where data points have membership degrees to multiple clusters
- Iterative optimization to minimize cost function
- Tested on multiple datasets with visualization

**Key Features:**
- **Soft Clustering:** Each data point belongs to multiple clusters with different degrees
- **Elbow Method:** Automatic determination of optimal cluster count
- **Visualization:** Color gradients showing fuzzy membership degrees
- **Multiple Datasets:** Tested on 4 different datasets
- **Iterative Algorithm:** Cost function minimization

**Technologies:** Python, NumPy, Pandas, Matplotlib

📂 **[View Project Repository](https://github.com/zamirmehdi/Fuzzy_C-means)**

---

### 3. Handwritten Digit Recognition

Multi-layer neural network built from scratch to classify handwritten digits using the MNIST dataset.

**Description:**
- Implemented a multi-layered neural network model from scratch
- Trained to recognize and classify handwritten digits (0-9)
- Uses backpropagation algorithm for training
- No deep learning frameworks - pure NumPy implementation

**Key Features:**
- **Dataset:** MNIST (60,000 training images + 10,000 test images)
- **Architecture:** Multi-layer feedforward neural network
- **Training:** Backpropagation with gradient descent
- **Implementation:** Built from scratch without frameworks
- **Stages:** Step-by-step implementation (step5, step6-1, step6-2, step6-3)

**Technologies:** Python, NumPy, Matplotlib, PIL (Pillow)

📂 **[View Project Repository](https://github.com/zamirmehdi/Handwritten-Digit-Recognition)**

## 🗂️ Repository Structure

```
Computational-Intelligence-Course/
├── Evolutionary-AI-Game-Project/
│   ├── docs/
│   │   ├── Instruction.pdf          # Project specification (Persian)
│   │   └── Report.pdf               # Implementation report (Persian)
│   ├── src/
│   │   ├── game.py                  # Main game implementation
│   │   ├── player.py                # Player agent class
│   │   ├── evolution.py             # Genetic algorithm implementation
│   │   ├── nn.py                    # Neural network (feedforward)
│   │   ├── config.py                # Configuration settings
│   │   ├── util.py                  # Utility functions
│   │   └── box_list.py              # Obstacle management
│   └── checkpoint/                  # Saved evolution states
├── Fuzzy_C-means/                   # External repository (submodule)
└── Handwritten-Digit-Recognition/   # External repository (submodule)
```

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.7+** | Primary programming language |
| **NumPy** | Numerical computing and matrix operations |
| **Matplotlib** | Data visualization and plotting |
| **Pygame** | Game development framework |
| **Pandas** | Data manipulation and analysis |
| **PIL/Pillow** | Image processing |

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/zamirmehdi/Computational-Intelligence-Course.git
cd Computational-Intelligence-Course
```

2. Install required packages:
```bash
pip install numpy matplotlib pandas pygame pillow
```

### Running the Projects

**Evolutionary AI Game:**
```bash
cd Evolutionary-AI-Game-Project/src
python game.py --mode helicopter --play True
```

**Fuzzy C-Means:**
```bash
cd Fuzzy_C-means
python main.py
```

**Handwritten Digit Recognition:**
```bash
cd Handwritten-Digit-Recognition/src
python main.py
```

Detailed instructions are available in each project's directory.

## ℹ️ Project Information

**Author:** Amirmehdi Zarrinnezhad  
**Course:** Computational Intelligence  
**University:** Amirkabir University of Technology (Tehran Polytechnic) - Spring 2021  
**Repository Link:** [Computational-Intelligence-Course](https://github.com/zamirmehdi/Computational-Intelligence-Course)

## 🔗 Related Courses

This repository is part of my coursework at Amirkabir University of Technology.

**Other Course Projects:**
- [Artificial Intelligence Course](https://github.com/zamirmehdi/Artificial-Intelligence-Course) - NLP, Search Algorithms, Pathfinding
  - [NLP Trigram Model](https://github.com/zamirmehdi/AI_Final_Project-NLP)
  - [Super Mario LRTA*](https://github.com/zamirmehdi/AI-Project-Super-Mario)
  - [Students Lineup](https://github.com/zamirmehdi/Artificial-Intelligence-Course/tree/main/Students-Lineup-Project)

## 📧 Contact

Questions or collaborations? Feel free to reach out!

**📧 Email:** amzarrinnezhad@gmail.com  
**🌐 GitHub:** [@zamirmehdi](https://github.com/zamirmehdi)

---


<p align="right">(<a href="#top">back to top</a>)</p>

<div align="center">

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

*Amirmehdi Zarrinnezhad*

</div>
