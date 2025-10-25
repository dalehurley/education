# Chapter 07: Code Snippets

Database migrations and seeding patterns.

## Files

### 1. `migration_example.py`

Alembic migration patterns and examples.

**Key Concepts:**

- Migration structure
- Upgrade/downgrade methods
- Add/modify columns
- Data migrations

### 2. `database_seeder.py`

Database seeding with factories.

**Run:**

```bash
python database_seeder.py
```

**Features:**

- Factory pattern for test data
- Faker for realistic data
- Seeder classes
- Bulk data generation

## Alembic Commands

```bash
# Initialize Alembic
alembic init alembic

# Create migration
alembic revision --autogenerate -m "Create users table"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Show current version
alembic current

# Show migration history
alembic history
```

## Laravel Comparison

| Alembic                | Laravel                        |
| ---------------------- | ------------------------------ |
| `alembic revision`     | `php artisan make:migration`   |
| `alembic upgrade head` | `php artisan migrate`          |
| `alembic downgrade -1` | `php artisan migrate:rollback` |
| Seeder classes         | `php artisan db:seed`          |
| Factory pattern        | `factory()` helper             |
