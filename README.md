# First PyTorch Project

A collection of small Python programs exploring **PyTorch and Deep Learning** through hands-on experiments.

This repository documents my learning journey, from basic tensor operations to neural networks and convolutional neural networks (CNNs).

## 📂 Project Structure

```text
first-pytorch-project/
├── main.py       # Initial PyTorch experiments
├── CNN.py        # Convolutional Neural Network experiments
├── README.md
└── pyproject.toml
```

Additional files and datasets may be generated locally when running the programs.

## 🧠 Topics Covered

- PyTorch tensors and tensor operations
- Neural network fundamentals
- Model training and evaluation
- Convolutional Neural Networks (CNNs)
- Computer vision and image classification
- Working with datasets using `torchvision`

*This repository evolves as I learn and experiment with new concepts.*

## 🛠️ Tech Stack

- **Python 3.11**
- **PyTorch** — Deep learning and tensor computation
- **Torchvision** — Computer vision datasets, transformations and models
- **uv** — Python project and dependency management

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sitylist94/first-pytorch-project.git
cd first-pytorch-project
```

### 2. Install dependencies

If the project dependencies are declared in `pyproject.toml`, install them with:

```bash
uv sync
```

If `torch` or `torchvision` is not yet declared, add the required dependencies:

```bash
uv add torch torchvision
```

### 3. Run an experiment

For the initial program:

```bash
uv run python main.py
```

For the CNN experiment:

```bash
uv run python CNN.py
```

## 🎯 Goals

- Build a solid understanding of Deep Learning fundamentals.
- Learn to implement and train neural networks with PyTorch.
- Explore computer vision through practical experiments.
- Track my progress through code and reproducible projects.

## 📌 Status

**Work in progress.** This repository is a learning project, and its contents will evolve as I explore more Deep Learning concepts.

## 📄 License

No license has been added yet.
