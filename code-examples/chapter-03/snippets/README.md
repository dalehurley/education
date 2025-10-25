# Chapter 03: Code Snippets

Reusable FastAPI REST API patterns.

## Files

### 1. `rest_api_patterns.py`

Complete CRUD REST API patterns.

**Run:**

```bash
uvicorn rest_api_patterns:app --reload
```

**Endpoints:**

- `GET /items` - List with pagination
- `GET /items/{id}` - Get single item
- `POST /items` - Create item
- `PUT /items/{id}` - Update item
- `DELETE /items/{id}` - Delete item
- `POST /items/bulk` - Bulk create

### 2. `response_models.py`

Different response types and status codes.

**Run:**

```bash
uvicorn response_models:app --reload
```

**Features:**

- JSON responses
- Custom status codes
- Custom headers
- Plain text responses
- Conditional responses

### 3. `query_parameters.py`

Advanced query parameter handling.

**Run:**

```bash
uvicorn query_parameters:app --reload
```

**Features:**

- Optional parameters
- Validation (min/max, patterns)
- List parameters
- Enums
- Aliases

## Testing

```bash
# Test REST API
curl http://localhost:8000/items
curl -X POST http://localhost:8000/items -d '{"name":"Test","price":9.99}'

# Test query parameters
curl "http://localhost:8000/search?q=python&limit=5"
curl "http://localhost:8000/filter?tags=python&tags=fastapi"
```
