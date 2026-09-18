# API Chat Query Flow Documentation

## Overview

This document explains the complete flow when a request is made to the `/api/chat/query` endpoint, tracing the path from the initial HTTP request to the final response.

## Endpoint Information

-   **URL**: `/api/chat/query`
-   **Method**: `POST`
-   **Content-Type**: `application/json`

## Sample Payload

```json
{
    "message": "What plan is in Rayudu Dhananjay Currently"
}
```

## Complete Request Flow

### 1. Entry Point: FastAPI Application (`app.py`)

```python
# File: app.py
# The main FastAPI application starts here
```

**Flow:**

-   FastAPI receives the POST request to `/api/chat/query`
-   Routes to the chat API module

---

### 2. API Route Handler (`api/chat_routes.py`)

```python
# File: api/chat_routes.py
@router.post("/query")
async def chat_query(request: ChatRequest)
```

**What happens:**

1. **Request Validation**: Pydantic validates the incoming JSON against `ChatRequest` model
2. **Extract Data**:
    - `message = request.message` ("What plan is in Rayudu Dhananjay Currently")
    - `user_context = request.user_context` (Doctor context)
3. **Agent Initialization**: Creates or gets existing `MedicalLangChainAgent`
4. **Set Context**: `agent.set_user_context(user_context)`
5. **Process Message**: `response = await agent.chat(message)`
6. **Return Response**: Sends back the agent's response

---

### 3. Medical LangChain Agent (`agents/medical_langchain_agent.py`)

```python
# File: agents/medical_langchain_agent.py
class MedicalLangChainAgent:
    async def chat(self, message: str) -> Dict[str, Any]
```

**What happens:**

1. **Add to History**: Stores user message in conversation history
2. **Truncate History**: Manages conversation memory (max 20 messages, 12000 tokens)
3. **LangChain Processing**: Uses OpenAI GPT-4o with custom prompt and tools
4. **Tool Selection**: AI decides which tool to use based on the message
5. **Execute Tools**: Runs selected tools and gets results
6. **Generate Response**: AI formulates final response

**Agent Configuration:**

-   **Model**: GPT-4o
-   **Temperature**: 0.1 (for consistency)
-   **Max Iterations**: 2
-   **Max Execution Time**: 120 seconds

---

### 4. Tool Selection and Execution (`tools/plan_tool.py`)

For the query "What plan is in Rayudu Dhananjay Currently", the agent selects:

```python
# File: tools/plan_tool.py
class PlanTool(BaseTool):
    def _run(self, patient_id=None, patient_name="Rayudu Dhananjaya", plan_type="current")
```

**What happens:**

1. **Role-Based Access Control**:
    - Checks user context (Doctor role = full access)
    - For patients: restricts to their own data only
2. **Parameter Processing**:
    - `patient_name = "Rayudu Dhananjaya"`
    - `plan_type = "current"`
3. **Database Manager Call**: `db_manager.get_current_active_plan(patient_name=patient_name)`

---

### 5. Database Manager (`dal/database.py`)

```python
# File: dal/database.py
class DatabaseManager:
    def get_current_active_plan(self, **kwargs)
```

**What happens:**

1. **Create Service Instance**: `plan_service = PlanService(self.db)`
2. **Delegate to Service**: `plan_service.get_current_active_plan(**kwargs)`
3. **Connection Management**: Handles database session lifecycle

---

### 6. Plan Service (`dal/services/plan_service.py`)

```python
# File: dal/services/plan_service.py
class PlanService(BaseService):
    def get_current_active_plan(self, patient_id=None, patient_name="Rayudu Dhananjaya")
```

**What happens:**

1. **Patient Name Resolution**:

    - Calls `self.find_patient_by_name_or_id(patient_id, patient_name)`
    - Searches Users table for matching name
    - Converts "Rayudu Dhananjaya" → patient_id (e.g., 132)

2. **Database Query**:

    ```sql
    SELECT my_plan.*, plan_master.*
    FROM my_plan
    JOIN plan_master ON my_plan.plan_id = plan_master.id
    WHERE my_plan.patient_id = 132
      AND my_plan.status = 1  -- Active
      AND my_plan.from_date <= NOW()
      AND (my_plan.to_date >= NOW() OR my_plan.to_date IS NULL)
    ORDER BY my_plan.from_date DESC
    LIMIT 1
    ```

3. **Result Processing**: Formats plan data into dictionary

---

### 7. Base Service (`dal/services/base_service.py`)

```python
# File: dal/services/base_service.py
class BaseService:
    def find_patient_by_name_or_id(self, patient_id=None, patient_name="Rayudu Dhananjaya")
```

**What happens:**

1. **Name Parsing**: Splits "Rayudu Dhananjaya" into ["Rayudu", "Dhananjaya"]
2. **Database Search**:
    ```sql
    SELECT * FROM users
    WHERE first_name ILIKE '%rayudu%'
      AND last_name ILIKE '%dhananjaya%'
    ```
3. **Return Patient ID**: Returns the matching user's ID

---

### 8. Database Models (`dal/models/`)

**Files involved:**

-   `dal/models/users.py` - User/Patient information
-   `dal/models/my_plan.py` - Patient's purchased plans
-   `dal/models/plan_master.py` - Plan definitions

**What happens:**

-   SQLAlchemy ORM executes the queries
-   Returns result objects with plan data

---

### 9. Response Journey Back

**9.1 Plan Service → Database Manager**

-   Returns plan dictionary or None

**9.2 Database Manager → Plan Tool**

-   Passes plan data to tool

**9.3 Plan Tool → LangChain Agent**

-   Formats response as JSON string
-   Example: `{"message": "No plans found for this patient", "has_plan": false}`

**9.4 LangChain Agent → API Route**

-   AI processes tool result
-   Generates natural language response
-   Example: "Rayudu Dhananjaya currently does not have any active plans."

**9.5 API Route → Client**

-   Returns JSON response:

```json
{
    "message": "Rayudu Dhananjaya currently does not have any active plans.",
    "metadata": {
        "agent_type": "Revival365AI Agent",
        "memory_messages": 2,
        "timestamp": "2025-08-05T10:30:00",
        "tools_available": 9,
        "response_length": 65
    }
}
```

---

## File Dependencies Summary

```
app.py
└── api/chat_routes.py
    └── agents/medical_langchain_agent.py
        └── tools/plan_tool.py
            └── dal/database.py
                └── dal/services/plan_service.py
                    └── dal/services/base_service.py
                        └── dal/models/
                            ├── users.py
                            ├── my_plan.py
                            └── plan_master.py
```

## Key Components by Layer

### **Presentation Layer**

-   `app.py` - FastAPI application
-   `api/chat_routes.py` - REST API endpoints

### **Business Logic Layer**

-   `agents/medical_langchain_agent.py` - AI agent orchestration
-   `tools/plan_tool.py` - Domain-specific tool logic

### **Data Access Layer**

-   `dal/database.py` - Database connection management
-   `dal/services/plan_service.py` - Business logic for plans
-   `dal/services/base_service.py` - Common database operations

### **Data Model Layer**

-   `dal/models/*.py` - SQLAlchemy ORM models

---

## Error Handling Flow

If any step fails:

1. **Database Error**: Service returns `{"error": "error_message"}`
2. **Tool Error**: Tool returns error string
3. **Agent Error**: Agent catches exception and returns error response
4. **API Error**: FastAPI returns 500 with error details

---

## Performance Considerations

1. **Database Connection**: Managed by context manager in DatabaseManager
2. **Memory Management**: Conversation history truncated to prevent token overflow
3. **Tool Timeout**: 120-second maximum execution time
4. **Agent Iterations**: Limited to 2 iterations to prevent infinite loops

---

## Security Features

1. **Role-Based Access Control**:
    - Patients can only access their own data
    - Medical staff can access all patient data
2. **Input Validation**: Pydantic models validate all inputs
3. **SQL Injection Prevention**: SQLAlchemy ORM parameterized queries
4. **Authentication Context**: User context passed through entire flow
