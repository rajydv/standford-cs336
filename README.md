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

Each assignment is self-contained with its own dependencies and environment setup. To work on a specific assignment:

### Prerequisites

- Python 3.8+
- `uv` package manager (recommended) or `pip`

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

3. **Create and activate a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using traditional methods
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

4. **Run tests to verify setup:**
   ```bash
   # Using uv
   uv run pytest
   
   # Or in activated environment
   pytest
   ```

## Assignment Details

### Assignment 1: Basics
- **Location:** `assignment1-basics/`
- **Topics:** Tokenization, transformer architecture, basic training
- **Key Files:**
  - `cs336_basics/tokenizer.py` - BPE tokenization implementation
  - `cs336_basics/model.py` - Transformer model components
  - `cs336_basics/optimizer.py` - AdamW optimizer implementation

### Assignment 2: Systems  
- **Location:** `assignment2-systems/`
- **Topics:** Distributed training, optimization techniques, benchmarking
- **Key Files:**
  - `cs336_systems/benchmarking.py` - Performance benchmarking tools

### Lecture Materials
- **Location:** `spring2025-lectures/`
- **Contents:** Code examples, reference implementations, and course materials

## Development Workflow

Each assignment follows the same pattern:

1. **Environment Setup:** Each assignment has its own `pyproject.toml` file
2. **Implementation:** Complete the TODO items in the provided skeleton code
3. **Testing:** Run the comprehensive test suite to validate your implementation
4. **Submission:** Use the provided submission scripts (e.g., `make_submission.sh`)

## Important Notes

- **Isolated Environments:** Each assignment uses its own virtual environment to avoid dependency conflicts
- **Testing:** Always run tests before considering an assignment complete
- **Data:** Some assignments include datasets in their `data/` directories
- **Documentation:** Each assignment includes detailed PDFs with instructions and theory

## Course Information

This is my personal implementation of Stanford's CS336 course assignments. The course covers:

- Tokenization and preprocessing
- Transformer architectures
- Optimization algorithms
- Distributed training systems
- Scaling laws and efficiency
- Advanced modeling techniques

## Usage

To run any specific file or test:

1. Navigate to the relevant assignment directory
2. Activate the environment (`uv sync` or activate your venv)
3. Run the desired Python files or tests

Example:
```bash
cd assignment1-basics
uv sync
uv run python cs336_basics/model.py
uv run pytest tests/test_model.py
```

## License

This repository contains educational implementations for learning purposes.