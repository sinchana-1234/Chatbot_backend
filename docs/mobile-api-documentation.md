# Revival Medical System - Chat/Query API Documentation

## Overview

This documentation provides everything the mobile development team needs to integrate with the Revival Medical System Chat/Query API. The API provides AI-powered medical data analysis and chat functionality with role-based access control.

## Base URL

```
https://your-domain.com/api/chat
```

## Authentication

All endpoints require authentication using Bearer tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Role-Based Access Control

The API supports two user roles:

-   **Patient (role_id: 1)**: Can only access their own medical data
-   **Medical Staff (role_id: 2+)**: Can access all patient data

---

## Endpoints

### 1. Health Check

**GET** `/health`

Simple health check to verify API availability.

**Response:**

```json
{
    "status": "healthy",
    "service": "Revival Medical System API",
    "version": "1.0.0"
}
```

### 2. API Information

**GET** `/`

Get general API information and available features.

**Response:**

```json
{
    "message": "Revival Medical System API",
    "version": "1.0.0",
    "features": ["Medical Data Analysis", "LangChain Agent", "Conversation Memory", "Patient Health Records"],
    "endpoints": ["/api/chat/query", "/api/chat/sessions"],
    "status": "active"
}
```

### 3. Active Sessions

**GET** `/sessions`

Get information about active chat sessions.

**Response:**

```json
{
    "active_sessions": 5,
    "session_ids": ["12345678...", "87654321..."]
}
```

### 1. Medical Query (Main Endpoint)

**POST** `/query`

Process medical queries with AI agent and get intelligent responses.

#### Request Body

```json
{
    "query": "What is my current blood pressure?",
    "sessionId": "optional-session-uuid",
    "patient_id": 123
}
```

#### Request Parameters

| Field        | Type    | Required | Description                                  |
| ------------ | ------- | -------- | -------------------------------------------- |
| `query`      | string  | ✅       | The medical question or query                |
| `sessionId`  | string  | ❌       | Session UUID for conversation continuity     |
| `patient_id` | integer | ❌       | Specific patient ID (for medical staff only) |

#### Response

```json
{
    "response": "Based on your latest readings, your blood pressure is 120/80 mmHg, which is within normal range.",
    "sessionId": "550e8400-e29b-41d4-a716-446655440000",
    "metadata": {
        "agent_type": "Revival365AI Agent",
        "memory_messages": 5,
        "timestamp": "2025-07-25T10:30:00.000Z",
        "tools_available": 8,
        "response_length": 127,
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "conversation_length": 5,
        "user_role": "Patient",
        "authorized_patient_id": 132
    },
    "user_context": {
        "user_id": 132,
        "role_name": "Patient",
        "can_access_all_patients": false,
        "authorized_patient_id": 132
    }
}
```

---

## Session Management

### What are Sessions?

Sessions maintain conversation context and memory across multiple queries. Each session has a unique UUID and preserves:

-   Conversation history
-   User context
-   Tool configurations

### Session Workflow

1. **First Query**: Send without `sessionId` to create a new session
2. **Follow-up Queries**: Use the returned `sessionId` to continue the conversation
3. **Session Persistence**: Sessions remain active for the duration of the user's interaction

### Example Session Flow

```javascript
// First query - creates new session
const response1 = await fetch("/api/chat/query", {
    method: "POST",
    headers: {
        Authorization: "Bearer <token>",
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        query: "What is my blood pressure?",
    }),
});

const data1 = await response1.json();
const sessionId = data1.sessionId; // Save this for subsequent queries

// Follow-up query - uses existing session
const response2 = await fetch("/api/chat/query", {
    method: "POST",
    headers: {
        Authorization: "Bearer <token>",
        "Content-Type": "application/json",
    },
    body: JSON.stringify({
        query: "What about yesterday?",
        sessionId: sessionId, // Use saved session ID
    }),
});
```

---

## Query Examples

### Patient Queries (Personal Data Only)

```json
{
    "query": "What is my current blood sugar level?"
}
```

```json
{
    "query": "Show my medication list"
}
```

```json
{
    "query": "What's my plan details?"
}
```

```json
{
    "query": "Show my food log for today"
}
```

### Medical Staff Queries (All Patient Data)

```json
{
    "query": "List all patients with high glucose readings",
    "patient_id": null
}
```

```json
{
    "query": "Show blood pressure for patient Rayudu",
    "patient_id": 132
}
```

```json
{
    "query": "Get treatment protocols for patient 111",
    "patient_id": 111
}
```

### Date-Based Queries

```json
{
    "query": "Show my blood pressure readings for this week"
}
```

```json
{
    "query": "List patients with high SpO2 this month"
}
```

```json
{
    "query": "What were my glucose readings yesterday?"
}
```

---

## Supported Medical Data Types

The API can query the following medical data:

| Data Type           | Description                       | Example Queries                                           |
| ------------------- | --------------------------------- | --------------------------------------------------------- |
| **Glucose**         | Blood sugar readings              | "What is my glucose level?", "Show high glucose patients" |
| **Blood Pressure**  | Systolic/Diastolic readings       | "My blood pressure today", "Patients with high BP"        |
| **Heart Rate**      | BPM measurements                  | "Current heart rate", "Heart rate trends"                 |
| **SpO2**            | Blood oxygen saturation           | "My oxygen levels", "Low SpO2 patients"                   |
| **Temperature**     | Body temperature                  | "Current temperature", "Fever patients"                   |
| **Sleep**           | Sleep duration and quality        | "My sleep analysis", "Sleep patterns"                     |
| **Medications**     | Current medications & supplements | "List my medications", "Current supplements"              |
| **Food Logs**       | Nutrition and dietary data        | "My food intake", "Nutrition logs"                        |
| **Treatment Plans** | Medical protocols and guidelines  | "My treatment plan", "Care instructions"                  |
| **Activity**        | Physical activity data            | "Activity levels", "Exercise data"                        |

---

## Error Handling

### HTTP Status Codes

| Code  | Description                                   |
| ----- | --------------------------------------------- |
| `200` | Success                                       |
| `400` | Bad Request (empty query, invalid parameters) |
| `401` | Unauthorized (invalid or missing token)       |
| `403` | Forbidden (insufficient permissions)          |
| `500` | Internal Server Error                         |

### Error Response Format

```json
{
    "detail": "Error message describing what went wrong"
}
```

### Common Errors

```json
// Empty query
{
  "detail": "Query cannot be empty"
}

// Invalid patient access
{
  "detail": "Access denied: Cannot access other patients' data"
}

// Agent error
{
  "detail": "Medical agent error: Tool execution failed"
}
```

---

## Mobile Implementation Examples

### React Native / JavaScript

```javascript
class MedicalChatAPI {
    constructor(baseUrl, authToken) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
        this.sessionId = null;
    }

    async sendQuery(query, patientId = null) {
        try {
            const response = await fetch(`${this.baseUrl}/api/chat/query`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${this.authToken}`,
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    query: query,
                    sessionId: this.sessionId,
                    patient_id: patientId,
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "API request failed");
            }

            const data = await response.json();

            // Save session ID for future queries
            if (data.sessionId) {
                this.sessionId = data.sessionId;
            }

            return data;
        } catch (error) {
            console.error("Medical query failed:", error);
            throw error;
        }
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/api/chat/health`);
        return response.json();
    }

    clearSession() {
        this.sessionId = null;
    }
}

// Usage example
const medicalAPI = new MedicalChatAPI("https://your-domain.com", "your-jwt-token");

// Send a query
medicalAPI
    .sendQuery("What is my blood pressure?")
    .then((response) => {
        console.log("AI Response:", response.response);
        console.log("Session ID:", response.sessionId);
    })
    .catch((error) => {
        console.error("Error:", error.message);
    });
```

### Flutter / Dart

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class MedicalChatAPI {
  final String baseUrl;
  final String authToken;
  String? sessionId;

  MedicalChatAPI(this.baseUrl, this.authToken);

  Future<Map<String, dynamic>> sendQuery(String query, {int? patientId}) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/api/chat/query'),
        headers: {
          'Authorization': 'Bearer $authToken',
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'query': query,
          'sessionId': sessionId,
          'patient_id': patientId,
        }),
      );

      if (response.statusCode != 200) {
        final error = jsonDecode(response.body);
        throw Exception(error['detail'] ?? 'API request failed');
      }

      final data = jsonDecode(response.body);

      // Save session ID for future queries
      if (data['sessionId'] != null) {
        sessionId = data['sessionId'];
      }

      return data;
    } catch (error) {
      print('Medical query failed: $error');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> healthCheck() async {
    final response = await http.get(Uri.parse('$baseUrl/api/chat/health'));
    return jsonDecode(response.body);
  }

  void clearSession() {
    sessionId = null;
  }
}

// Usage example
final medicalAPI = MedicalChatAPI('https://your-domain.com', 'your-jwt-token');

medicalAPI.sendQuery("What is my blood pressure?").then((response) {
  print('AI Response: ${response['response']}');
  print('Session ID: ${response['sessionId']}');
}).catchError((error) {
  print('Error: $error');
});
```

---

## Best Practices

### 1. Session Management

-   Always preserve and reuse `sessionId` for conversational flows
-   Clear sessions when starting new conversations or switching patients
-   Handle session expiration gracefully

### 2. Error Handling

-   Always check HTTP status codes
-   Implement retry logic for network failures
-   Show user-friendly error messages

### 3. Security

-   Never expose JWT tokens in logs or client-side storage
-   Use HTTPS for all API communications
-   Implement proper token refresh mechanisms

### 4. Performance

-   Cache session IDs locally
-   Implement request timeouts (recommended: 30-60 seconds)
-   Show loading indicators for long-running queries

### 5. User Experience

-   Provide typing indicators while processing queries
-   Show conversation history using session continuity
-   Implement offline mode with cached responses

---

## Testing

### API Testing with curl

```bash
# Health check
curl -X GET "https://your-domain.com/api/chat/health"

# Medical query
curl -X POST "https://your-domain.com/api/chat/query" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my blood pressure?",
    "sessionId": null,
    "patient_id": null
  }'
```

### Postman Collection

Import the following collection to test all endpoints:

```json
{
    "info": {
        "name": "Revival Medical Chat API",
        "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
    },
    "item": [
        {
            "name": "Health Check",
            "request": {
                "method": "GET",
                "header": [],
                "url": "{{baseUrl}}/api/chat/health"
            }
        },
        {
            "name": "Medical Query",
            "request": {
                "method": "POST",
                "header": [
                    {
                        "key": "Authorization",
                        "value": "Bearer {{authToken}}"
                    },
                    {
                        "key": "Content-Type",
                        "value": "application/json"
                    }
                ],
                "body": {
                    "mode": "raw",
                    "raw": "{\n  \"query\": \"What is my blood pressure?\",\n  \"sessionId\": null,\n  \"patient_id\": null\n}"
                },
                "url": "{{baseUrl}}/api/chat/query"
            }
        }
    ],
    "variable": [
        {
            "key": "baseUrl",
            "value": "https://your-domain.com"
        },
        {
            "key": "authToken",
            "value": "your-jwt-token-here"
        }
    ]
}
```

---

## Support

For technical support or questions about this API:

-   **Email**: dev-support@revival-medical.com
-   **Documentation**: [API Docs](https://docs.revival-medical.com)
-   **Status Page**: [Status](https://status.revival-medical.com)

---

## Changelog

### v1.0.0 (Current)

-   Initial release
-   Role-based access control
-   Session management
-   AI-powered medical queries
-   Comprehensive medical data support
