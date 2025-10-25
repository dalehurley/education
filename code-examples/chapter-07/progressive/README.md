# Chapter 07: Task Manager v7 - Migrations

**Progressive Build**: Adds Alembic migrations to v6

## 🆕 What's New

- ✅ **Alembic Migrations**: Version control for DB schema
- ✅ **Auto-generate**: Detect model changes
- ✅ **Seeders**: Populate test data
- ✅ **Migration History**: Track schema changes

## 🚀 Setup

```bash
cd code-examples/chapter-07/progressive
pip install -r requirements.txt

# Initialize Alembic (already done)
# alembic init alembic

# Generate migration
alembic revision --autogenerate -m "Initial schema"

# Run migration
alembic upgrade head

# Seed data
python seed.py

# Run app
uvicorn task_manager_v7_migrations:app --reload
```

## 📋 Migration Commands

```bash
# Create migration (auto-detect changes)
alembic revision --autogenerate -m "Add new column"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show migration history
alembic history

# Show current version
alembic current
```

## 🌱 Seeding

```bash
python seed.py
```

Creates:

- 2 test users (alice, bob)
- 5 sample tasks

## 🎓 Key Concepts

**Migrations**: Like Laravel migrations
**Autogenerate**: Detects model changes
**Seeders**: Test data population
**Version Control**: Track schema evolution
