# Stanford CS336: Language Modeling from Scratch (Spring 2025)

This repository contains my solutions and implementations for Stanford's CS336 course on Language Modeling from Scratch. The course covers the fundamentals of building language models, from basic components to advanced systems.

## Repository Structure

The repository is organized into separate directories for each assignment, with each containing its own complete project setup:

```
├── assignment1-basics/          # Fundamentals: tokenization, models, optimization
├── assignment2-systems/         # Systems: parallelism, optimization, benchmarking  
├── spring2025-lectures/         # Lecture materials and code examples
└── README.md                    # This file
```

## Getting Started

Each assignment is self-contained with its own dependencies and environment setup. We use `uv` for dependency management to ensure reproducibility, portability, and ease of use.

### Prerequisites

- Python 3.8+
- `uv` package manager - Install from [here](https://github.com/astral-sh/uv) or run `pip install uv`/`brew install uv`

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rajydv/standford-cs336.git
   cd stanford-cs336
   ```

2. **Navigate to the specific assignment:**
   ```bash
   cd assignment1-basics  # or assignment2-systems, etc.
   ```

3. **Set up environment and run code:**
   ```bash
   # Using uv (recommended) - automatically manages environment
   uv run <python_file_path>
   uv run pytest  # Run tests
   ```
