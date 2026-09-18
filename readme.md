# HospitalChatApp: Medical ChatGPT API Integration with Pinecone

**Note:** This branch is dedicated to the hospital/medical project.

This project provides a robust API for processing medical user queries using OpenAI's GPT models and Pinecone for vector-based semantic search. The backend is built with FastAPI and SQLAlchemy ORM, supporting modular, maintainable code and scalable medical data management for hospital use cases.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Setup Instructions](#setup-instructions)
5. [Environment Variables](#environment-variables)
6. [Database Models](#database-models)
7. [Running the API Server](#running-the-api-server)
8. [Testing the API](#testing-the-api)
9. [Development & Extending](#development--extending)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## Features

-   Natural language chat API using OpenAI GPT
-   Semantic document search with Pinecone
-   Modular SQLAlchemy ORM models (medical readings, medications, users, etc.)
-   FastAPI endpoints for chat, document, and data queries
-   Easily extensible tool and agent architecture
-   Robust database connection and session management

---

## Project Structure

```
Python/
├── dal/
│   ├── database.py         # DB connection, session, and manager logic
│   └── models/
│       ├── base.py         # Shared SQLAlchemy Base
│       ├── glucose_readings.py
│       ├── activity_readings.py
│       ├── blood_pressure_readings.py
│       ├── ... (other tables)
│       └── users.py
├── api/                    # API logic (e.g., rag_query_logic.py)
├── tools/                  # Tool classes for agent actions
├── agents/                 # Agent logic and orchestration
├── start.py                # FastAPI app entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
└── ...
```

---

## Prerequisites

Ensure you have the following installed:

-   Python 3.8 or higher
-   pip (Python package manager)
-   [Pinecone](https://www.pinecone.io/) account and API key
-   [OpenAI](https://platform.openai.com/) account and API key
-   MySQL database (cloud or local)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SchoolChatApp/code/Python
```

### 2. Create and Activate a Virtual Environment

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the `Python/` directory with the following content:

```properties
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"
INDEX_NAME="your-pinecone-index-name"
PINECONE_REGION="your-pinecone-region"
MYSQL_HOST="your-mysql-host"
MYSQL_PORT="3306"
MYSQL_DATABASE="your-db-name"
MYSQL_USERNAME="your-db-username"
MYSQL_PASSWORD="your-db-password"
```

---

## Database Models

All SQLAlchemy ORM models are now organized under `dal/models/`, one file per table. To add a new table, create a new file in this folder and import `Base` from `base.py`.

---

## Running the API Server

Start the FastAPI server using Uvicorn:

```bash
# Activate your virtual environment first
.venv\Scripts\activate
uvicorn start:app --reload
# Or specify a port
uvicorn start:app --reload --port 8080
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Testing the API

You can test the API using Postman, curl, or any HTTP client. Example:

```bash
curl -X POST "http://127.0.0.1:8000/api/query" -H "Content-Type: application/json" -d '{"userQuery": "Your query here"}'
```

Interactive API docs are available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Development & Extending

-   To add new database tables: create a new file in `dal/models/` and update imports in `dal/database.py`.
-   To add new tools or agent logic: add to `tools/` or `agents/` and register in the main app.
-   For document search or RAG: see `api/rag_query_logic.py` and related tools.

---

## Troubleshooting

-   Ensure your `.env` file is present and correct.
-   If you see DB connection errors, check your MySQL credentials and network.
-   For Pinecone/OpenAI errors, verify your API keys and region.
-   Use the logs for debugging (see `logging` setup in code).

---

## License

This project is for internal/educational use. See LICENSE file if present.

## remove cache

find . -name "**pycache**" -type d -exec rm -rf {} + 2>/dev/null || true
