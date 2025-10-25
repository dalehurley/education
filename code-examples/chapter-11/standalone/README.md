# Chapter 11: Multi-tenant SaaS API

Complete OAuth2 authentication with multi-tenancy.

## 🎯 Features

- ✅ OAuth2 password flow
- ✅ JWT access and refresh tokens
- ✅ Role-based access control (RBAC)
- ✅ Multi-tenant data isolation
- ✅ Password hashing with bcrypt

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn multitenant_saas:app --reload
```

## 👥 Test Users

- **user1** / password123 (Tenant 1, User role)
- **admin1** / admin123 (Tenant 1, Admin role)
- **user2** / password123 (Tenant 2, User role)

## 💡 Usage

```bash
# Login
curl -X POST "http://localhost:8000/token" \
  -d "username=user1&password=password123"

# Use access token
curl "http://localhost:8000/me" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

## 🎓 Key Concepts

**OAuth2**: Industry standard for authorization
**JWT**: Stateless authentication tokens
**RBAC**: Role-based permissions
**Multi-tenancy**: Data isolation per tenant
