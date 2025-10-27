# Xuve

A RAG system for searching and retrieving information from movies data.

## Description

Xuve is a command-line tool that allows you to search for movies based on a keyword query. It utilizes a Retrieval-Augmented Generation (RAG) system to provide relevant results from a dataset of movies.

## Features

*   Keyword-based search for movies.
*   Command-line interface for easy interaction.
*   Built with Python and the NLTK library.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/xonoxc/xuve.git
    cd xuve
    ```

2.  **Create and activate a virtual environment:**

    This project recommends using `uv` to manage the virtual environment.

    ```bash
    uv venv
    source .venv/bin/activate
    ```

    Alternatively, you can use Python's built-in `venv` module:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**

    This project uses `uv` for package management. If you don't have it, install it first (`pip install uv`).

    ```bash
    uv sync
    ```

## Usage

To search for movies, use the `search` command followed by your query:

```bash
python -m cli.main search "your query here"
```

### Example

```bash
python -m cli.main search "a movie about a spy"
```

## Project Structure

```
.
├── cli/                # Command-line interface logic
├── config/             # Configuration files
├── data/               # Data files
├── decors/             # Decorators
├── lib/                # Core application logic
│   ├── indexes/
├── typedicts/          # Type definitions
├── .gitignore
├── .python-version
├── pyproject.toml
├── pyrightconfig.json
├── README.md
└── uv.lock
```

## Dependencies

*   [nltk](https://www.nltk.org/): The Natural Language Toolkit, used for processing text data.
*   [sentence_transformers](https://huggingface.co/sentence-transformers): The Natural Language Toolkit, used for processing text data.

## Model 
    - all-MiniLM-L6-v2
        

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
