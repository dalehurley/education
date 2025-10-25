# Chapter 11: Task Manager v11 - OAuth & Multi-tenancy

**Progressive Build**: Adds OAuth2 and workspaces to v10

## 🆕 What's New

- ✅ **OAuth2**: Standard OAuth2 password flow
- ✅ **Refresh Tokens**: Long-lived sessions
- ✅ **Multi-tenancy**: Workspace isolation
- ✅ **RBAC**: Role-based access control

## 🚀 Usage

```bash
# Register
curl -X POST "http://localhost:8000/auth/register" \
  -d '{"username": "john", "email": "john@example.com", "password": "secret"}'

# Login (OAuth2)
curl -X POST "http://localhost:8000/auth/token" \
  -d "username=john&password=secret"

# Create workspace
curl -X POST "http://localhost:8000/workspaces" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"name": "My Team", "slug": "my-team"}'
```
