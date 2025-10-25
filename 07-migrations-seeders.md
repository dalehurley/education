# Chapter 07: Migrations & Seeders

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Alembic for database migrations
- Create and run migrations safely
- Handle migration rollbacks and reversibility
- Seed your database with test data
- Manage database versions across environments
- Follow migration best practices for production
- Troubleshoot common migration issues

## 🔄 Laravel vs Alembic

| Feature          | Laravel                        | Alembic                           |
| ---------------- | ------------------------------ | --------------------------------- |
| Tool             | Built-in migrations            | Alembic (separate package)        |
| Create migration | `php artisan make:migration`   | `alembic revision --autogenerate` |
| Run migrations   | `php artisan migrate`          | `alembic upgrade head`            |
| Rollback         | `php artisan migrate:rollback` | `alembic downgrade -1`            |
| Status           | `php artisan migrate:status`   | `alembic current`                 |
| History          | `php artisan migrate:status`   | `alembic history`                 |
| Seed             | `php artisan db:seed`          | Custom scripts                    |
| Fresh            | `php artisan migrate:fresh`    | Drop + upgrade                    |
| Show pending     | N/A                            | `alembic heads`                   |
| Stamp version    | N/A                            | `alembic stamp head`              |

## 📚 Core Concepts

### 1. Installing and Setting Up Alembic

```bash
pip install alembic
```

**Initialize Alembic:**

```bash
cd /path/to/your/project
alembic init alembic
```

This creates:

```
alembic/
├── versions/          # Migration files
├── env.py            # Alembic environment config
├── script.py.mako    # Migration template
└── README
alembic.ini           # Alembic config file
```

### 2. Configuration

**Laravel:**

```php
<?php
// config/database.php
return [
    'default' => env('DB_CONNECTION', 'mysql'),
    'connections' => [
        'mysql' => [
            'host' => env('DB_HOST', '127.0.0.1'),
            // ...
        ]
    ]
];
```

**Alembic:**

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your models and database config
from app.core.database import Base
from app.core.config import settings
from app.models import user, post  # Import all models!

# Alembic Config object
config = context.config

# Set database URL from your settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Model metadata for auto-generation
target_metadata = Base.metadata

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Note for SQLAlchemy 2.0+:**

If using SQLAlchemy 2.0 with async, modify `run_migrations_online()`:

```python
async def run_async_migrations():
    """Run migrations in async mode (SQLAlchemy 2.0+)"""
    from sqlalchemy.ext.asyncio import create_async_engine

    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    import asyncio
    asyncio.run(run_async_migrations())
```

```ini
# alembic.ini
[alembic]
script_location = alembic
prepend_sys_path = .

# Can be overridden in env.py
sqlalchemy.url = sqlite:///./app.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

### 3. Creating Migrations

**Laravel:**

```bash
php artisan make:migration create_users_table
php artisan make:migration add_role_to_users_table
```

**Alembic:**

```bash
# Auto-generate migration from models
alembic revision --autogenerate -m "create users table"

# Manual migration
alembic revision -m "add role to users"
```

**Auto-generated Migration:**

```python
# alembic/versions/001_create_users_table.py
"""create users table

Revision ID: abc123
Revises:
Create Date: 2024-01-01 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('age', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
```

**Manual Migration:**

```python
# alembic/versions/002_add_role_to_users.py
"""add role to users

Revision ID: def456
Revises: abc123
Create Date: 2024-01-02 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

revision = 'def456'
down_revision = 'abc123'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('users', sa.Column('role', sa.String(50), nullable=True))

    # Set default value for existing rows
    op.execute("UPDATE users SET role = 'user' WHERE role IS NULL")

    # Make column non-nullable
    op.alter_column('users', 'role', nullable=False)

def downgrade():
    op.drop_column('users', 'role')
```

**⚠️ Autogenerate Limitations:**

Alembic's `--autogenerate` is powerful but has limitations. It **cannot** detect:

- Table or column renames (sees them as drop + add)
- Changes to constraints without explicit names
- Enum value changes (PostgreSQL)
- Index type changes

**Always review auto-generated migrations before running them!**

### 4. Running Migrations

**Laravel:**

```bash
php artisan migrate                # Run migrations
php artisan migrate:rollback       # Rollback last batch
php artisan migrate:reset          # Rollback all
php artisan migrate:fresh          # Drop all + migrate
php artisan migrate:refresh        # Reset + migrate
php artisan migrate:status         # Show status
```

**Alembic:**

```bash
# Run all pending migrations
alembic upgrade head

# Upgrade to specific revision
alembic upgrade abc123

# Rollback one migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade abc123

# Rollback all
alembic downgrade base

# Show current version
alembic current

# Show migration history
alembic history

# Show pending migrations
alembic heads
```

### 5. Common Migration Operations

**Create Table:**

```python
def upgrade():
    op.create_table(
        'posts',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

def downgrade():
    op.drop_table('posts')
```

**Add Column:**

```python
def upgrade():
    op.add_column('users', sa.Column('phone', sa.String(20), nullable=True))

def downgrade():
    op.drop_column('users', 'phone')
```

**Modify Column:**

```python
def upgrade():
    op.alter_column('users', 'name',
                    existing_type=sa.String(100),
                    type_=sa.String(255),
                    nullable=False)

def downgrade():
    op.alter_column('users', 'name',
                    existing_type=sa.String(255),
                    type_=sa.String(100),
                    nullable=True)
```

**Add Index:**

```python
def upgrade():
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    op.drop_index('ix_users_email', table_name='users')
```

**Add Composite Index:**

```python
def upgrade():
    # Composite index on multiple columns
    op.create_index('ix_posts_user_created', 'posts', ['user_id', 'created_at'])

    # Unique composite index
    op.create_index('ix_votes_unique', 'votes', ['user_id', 'post_id'], unique=True)

def downgrade():
    op.drop_index('ix_votes_unique', table_name='votes')
    op.drop_index('ix_posts_user_created', table_name='posts')
```

**Add Foreign Key:**

```python
def upgrade():
    op.create_foreign_key(
        'fk_posts_user_id',
        'posts', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )

def downgrade():
    op.drop_constraint('fk_posts_user_id', 'posts', type_='foreignkey')
```

**Rename Column:**

```python
def upgrade():
    op.alter_column('users', 'name', new_column_name='full_name')

def downgrade():
    op.alter_column('users', 'full_name', new_column_name='name')
```

**Add Unique Constraint:**

```python
def upgrade():
    op.create_unique_constraint('uq_users_email', 'users', ['email'])

def downgrade():
    op.drop_constraint('uq_users_email', 'users', type_='unique')
```

**Add Check Constraint:**

```python
def upgrade():
    op.create_check_constraint(
        'ck_users_age_positive',
        'users',
        'age > 0'
    )

def downgrade():
    op.drop_constraint('ck_users_age_positive', 'users', type_='check')
```

**Add JSON Column:**

```python
def upgrade():
    # PostgreSQL JSONB
    op.add_column('users', sa.Column('metadata', sa.dialects.postgresql.JSONB, nullable=True))

    # Or generic JSON for cross-database compatibility
    op.add_column('users', sa.Column('settings', sa.JSON, nullable=True))

def downgrade():
    op.drop_column('users', 'settings')
    op.drop_column('users', 'metadata')
```

**Add Enum Type (PostgreSQL):**

```python
def upgrade():
    # Create enum type
    op.execute("CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended')")

    # Add column using enum
    op.add_column('users',
        sa.Column('status', sa.Enum('active', 'inactive', 'suspended', name='user_status'))
    )

def downgrade():
    op.drop_column('users', 'status')
    op.execute('DROP TYPE user_status')
```

### 6. Data Migrations

**Laravel:**

```php
<?php
use Illuminate\Database\Migrations\Migration;

class UpdateUserRoles extends Migration
{
    public function up()
    {
        DB::table('users')
            ->where('is_admin', true)
            ->update(['role' => 'admin']);
    }

    public function down()
    {
        DB::table('users')
            ->where('role', 'admin')
            ->update(['is_admin' => true, 'role' => null]);
    }
}
```

**Alembic:**

```python
"""update user roles

Revision ID: ghi789
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column

def upgrade():
    # Create a temporary table for data manipulation
    users = table('users',
        column('id', sa.Integer),
        column('is_admin', sa.Boolean),
        column('role', sa.String)
    )

    # Update existing data
    op.execute(
        users.update()
        .where(users.c.is_admin == True)
        .values(role='admin')
    )

    op.execute(
        users.update()
        .where(users.c.is_admin == False)
        .values(role='user')
    )

def downgrade():
    users = table('users',
        column('role', sa.String),
        column('is_admin', sa.Boolean)
    )

    op.execute(
        users.update()
        .where(users.c.role == 'admin')
        .values(is_admin=True)
    )
```

### 7. Database Seeding

**Laravel:**

```php
<?php
// database/seeders/DatabaseSeeder.php
class DatabaseSeeder extends Seeder
{
    public function run()
    {
        User::factory(10)->create();

        User::create([
            'name' => 'Admin',
            'email' => 'admin@example.com',
            'password' => Hash::make('password'),
        ]);
    }
}
```

**Python/FastAPI:**

```python
# app/db/seed.py
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import AsyncSessionLocal, engine, Base
from app.models.user import User
from app.models.post import Post

async def seed_database():
    """Seed the database with initial data"""

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async with AsyncSessionLocal() as session:
        # Check if already seeded
        from sqlalchemy import select
        result = await session.execute(select(User))
        if result.scalars().first():
            print("Database already seeded")
            return

        # Create users
        users = [
            User(name="Admin", email="admin@example.com", role="admin", age=30),
            User(name="John Doe", email="john@example.com", role="user", age=25),
            User(name="Jane Smith", email="jane@example.com", role="user", age=28),
        ]

        session.add_all(users)
        await session.commit()

        # Refresh to get IDs
        for user in users:
            await session.refresh(user)

        # Create posts
        posts = [
            Post(
                title="First Post",
                content="This is the first post",
                published=True,
                user_id=users[0].id
            ),
            Post(
                title="Second Post",
                content="This is the second post",
                published=True,
                user_id=users[1].id
            ),
            Post(
                title="Draft Post",
                content="This is a draft",
                published=False,
                user_id=users[1].id
            ),
        ]

        session.add_all(posts)
        await session.commit()

        print("Database seeded successfully!")

if __name__ == "__main__":
    asyncio.run(seed_database())
```

**Run seeder:**

```bash
python -m app.db.seed
```

**Synchronous Alternative (without async):**

```python
# app/db/seed_sync.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.core.database import Base
from app.models.user import User
from app.models.post import Post

def seed_database_sync():
    """Seed the database with initial data (synchronous)"""

    # Create engine and session
    engine = create_engine(settings.DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    session = SessionLocal()

    try:
        # Check if already seeded
        if session.query(User).first():
            print("Database already seeded")
            return

        # Create users
        admin = User(name="Admin", email="admin@example.com", role="admin", age=30)
        john = User(name="John Doe", email="john@example.com", role="user", age=25)
        jane = User(name="Jane Smith", email="jane@example.com", role="user", age=28)

        session.add_all([admin, john, jane])
        session.commit()

        # Create posts
        posts = [
            Post(title="First Post", content="Content here", published=True, user_id=admin.id),
            Post(title="Second Post", content="More content", published=True, user_id=john.id),
            Post(title="Draft", content="Draft content", published=False, user_id=john.id),
        ]

        session.add_all(posts)
        session.commit()

        print("Database seeded successfully!")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    seed_database_sync()
```

### 8. Factory Pattern for Test Data

```python
# app/db/factories.py
from faker import Faker
from app.models.user import User
from app.models.post import Post
import random

fake = Faker()

def user_factory(**kwargs):
    """Generate fake user data"""
    defaults = {
        "name": fake.name(),
        "email": fake.email(),
        "age": random.randint(18, 80),
        "role": random.choice(["user", "admin"]),
        "is_active": True
    }
    defaults.update(kwargs)
    return User(**defaults)

def post_factory(user_id: int = None, **kwargs):
    """Generate fake post data"""
    defaults = {
        "title": fake.sentence(),
        "content": fake.text(500),
        "published": random.choice([True, False]),
        "user_id": user_id or 1
    }
    defaults.update(kwargs)
    return Post(**defaults)

# Usage in seed script
async def seed_with_factories():
    async with AsyncSessionLocal() as session:
        # Create 10 users
        users = [user_factory() for _ in range(10)]
        session.add_all(users)
        await session.commit()

        # Create 50 posts
        for user in users:
            await session.refresh(user)
            posts = [post_factory(user_id=user.id) for _ in range(5)]
            session.add_all(posts)

        await session.commit()
```

Install Faker:

```bash
pip install faker
```

### 9. Complete Migration Workflow

**Create models:**

```python
# app/models/category.py
from sqlalchemy import Column, Integer, String
from app.core.database import Base

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)

# app/models/post.py - Update to add category
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class Post(Base):
    # ... existing columns
    category_id = Column(Integer, ForeignKey('categories.id'), nullable=True)

    category = relationship("Category")
```

**Generate migration:**

```bash
alembic revision --autogenerate -m "add categories table and relationship"
```

**Review and edit migration:**

```python
# alembic/versions/xxx_add_categories.py
def upgrade():
    # Create categories table
    op.create_table(
        'categories',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('slug')
    )

    # Add category_id to posts
    op.add_column('posts', sa.Column('category_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_posts_category', 'posts', 'categories',
                         ['category_id'], ['id'])

    # Seed default categories
    op.execute("""
        INSERT INTO categories (name, slug) VALUES
        ('Technology', 'technology'),
        ('Business', 'business'),
        ('Lifestyle', 'lifestyle')
    """)

def downgrade():
    op.drop_constraint('fk_posts_category', 'posts', type_='foreignkey')
    op.drop_column('posts', 'category_id')
    op.drop_table('categories')
```

**Run migration:**

```bash
alembic upgrade head
```

### 10. Migration Helper CLI

Create a helper script for common migration tasks:

```python
# scripts/migrate.py
"""
Migration helper script for common Alembic operations.

Usage:
    python scripts/migrate.py create "add user status field"
    python scripts/migrate.py up
    python scripts/migrate.py down
    python scripts/migrate.py status
    python scripts/migrate.py fresh
"""

import sys
import subprocess
from pathlib import Path

def run_command(cmd: list[str], description: str = ""):
    """Run a shell command and handle errors"""
    if description:
        print(f"🔧 {description}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e.stderr}")
        return False

def create_migration(message: str):
    """Create a new migration"""
    run_command(
        ["alembic", "revision", "--autogenerate", "-m", message],
        f"Creating migration: {message}"
    )

def migrate_up():
    """Run all pending migrations"""
    run_command(["alembic", "upgrade", "head"], "Running migrations...")

def migrate_down(steps: int = 1):
    """Rollback migrations"""
    run_command(
        ["alembic", "downgrade", f"-{steps}"],
        f"Rolling back {steps} migration(s)..."
    )

def show_status():
    """Show current migration status"""
    print("📊 Current Migration Status:")
    run_command(["alembic", "current"], "")
    print("\n📋 Migration History:")
    run_command(["alembic", "history"], "")

def migrate_fresh():
    """Drop all tables and run migrations from scratch"""
    response = input("⚠️  This will DROP ALL TABLES. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return

    run_command(["alembic", "downgrade", "base"], "Dropping all tables...")
    run_command(["alembic", "upgrade", "head"], "Running all migrations...")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "create":
        if len(sys.argv) < 3:
            print("Error: Migration message required")
            print('Usage: python scripts/migrate.py create "message"')
            sys.exit(1)
        create_migration(sys.argv[2])

    elif command == "up":
        migrate_up()

    elif command == "down":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        migrate_down(steps)

    elif command == "status":
        show_status()

    elif command == "fresh":
        migrate_fresh()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Create new migration
python scripts/migrate.py create "add user status field"

# Run migrations
python scripts/migrate.py up

# Rollback one migration
python scripts/migrate.py down

# Rollback multiple migrations
python scripts/migrate.py down 3

# Show status
python scripts/migrate.py status

# Fresh migration (drops all and re-runs)
python scripts/migrate.py fresh
```

## ⚡ Best Practices

### 1. Always Make Migrations Reversible

Every migration should have a proper `downgrade()` function:

```python
# ❌ Bad - No way to undo
def downgrade():
    pass

# ✅ Good - Fully reversible
def downgrade():
    op.drop_constraint('fk_posts_category', 'posts', type_='foreignkey')
    op.drop_column('posts', 'category_id')
```

### 2. Test Migrations in Development First

```bash
# Test the full cycle
alembic upgrade head    # Apply migration
alembic downgrade -1    # Rollback
alembic upgrade head    # Re-apply

# Check the database state after each step
```

### 3. Use Transactions for Data Migrations

```python
def upgrade():
    # Use batch operations for large datasets
    from sqlalchemy import table, column
    users = table('users', column('email'))

    # This runs in a transaction
    op.execute(
        users.update()
        .where(users.c.email == None)
        .values(email='noemail@example.com')
    )
```

### 4. Backup Before Production Migrations

```bash
# Always backup before running migrations in production
pg_dump -U username dbname > backup_$(date +%Y%m%d_%H%M%S).sql

# Then run migrations
alembic upgrade head
```

### 5. Handle Large Data Sets Carefully

For migrations affecting millions of rows:

```python
def upgrade():
    # Bad - Locks entire table
    op.execute("UPDATE users SET updated_at = NOW()")

    # Good - Batch updates
    op.execute("""
        UPDATE users
        SET updated_at = NOW()
        WHERE id IN (SELECT id FROM users LIMIT 1000)
    """)
    # Repeat in chunks with a script
```

### 6. Name Constraints Explicitly

```python
# ✅ Good - Named constraints can be easily dropped
op.create_foreign_key(
    'fk_posts_user_id',  # Explicit name
    'posts', 'users',
    ['user_id'], ['id']
)

# ❌ Bad - Auto-generated names vary by database
op.create_foreign_key(
    None,  # Database generates name
    'posts', 'users',
    ['user_id'], ['id']
)
```

### 7. Don't Mix Schema and Data Changes

```python
# ❌ Bad - Mixed concerns
def upgrade():
    op.add_column('users', sa.Column('status', sa.String(20)))
    op.execute("UPDATE users SET status = 'active'")

# ✅ Good - Separate migrations
# Migration 1: Add column (schema)
def upgrade():
    op.add_column('users', sa.Column('status', sa.String(20), nullable=True))

# Migration 2: Populate data
def upgrade():
    op.execute("UPDATE users SET status = 'active' WHERE status IS NULL")
    op.alter_column('users', 'status', nullable=False)
```

### 8. Version Control Best Practices

- **Commit migrations with code changes** that require them
- **Never edit applied migrations** - create a new one to fix issues
- **Use descriptive migration messages**: `alembic revision -m "add user role field"`
- **Review auto-generated migrations** before committing

### 9. Environment-Specific Considerations

```python
# alembic/env.py - Handle different environments
import os
from app.core.config import settings

config = context.config

# Override from environment
if os.getenv("DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", os.getenv("DATABASE_URL"))
else:
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
```

### 10. Use Batch Operations for SQLite

SQLite has limited ALTER TABLE support. Use batch mode:

```python
def upgrade():
    with op.batch_alter_table('users') as batch_op:
        batch_op.add_column(sa.Column('role', sa.String(50)))
        batch_op.create_index('ix_users_role', ['role'])
```

## 🔧 Troubleshooting Common Issues

### Issue 1: "Can't locate revision identified by 'xyz'"

**Cause:** Migration file missing or version mismatch

**Solution:**

```bash
# Check current version
alembic current

# Check history
alembic history

# If database is ahead of code, stamp to match
alembic stamp head
```

### Issue 2: "Target database is not up to date"

**Cause:** Database has unapplied migrations

**Solution:**

```bash
# See what's pending
alembic heads

# Apply pending migrations
alembic upgrade head
```

### Issue 3: "Multiple heads detected"

**Cause:** Multiple branches in migration history

**Solution:**

```bash
# View the branches
alembic heads

# Merge the branches
alembic merge heads -m "merge branches"
```

### Issue 4: "Table already exists"

**Cause:** Migration trying to create existing table

**Solution:**

```python
def upgrade():
    # Check if table exists first
    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if 'users' not in inspector.get_table_names():
        op.create_table('users', ...)
```

### Issue 5: Foreign Key Constraint Failure

**Cause:** Data violates new constraint

**Solution:**

```python
def upgrade():
    # Clean up orphaned records first
    op.execute("""
        DELETE FROM posts
        WHERE user_id NOT IN (SELECT id FROM users)
    """)

    # Then add constraint
    op.create_foreign_key('fk_posts_user', 'posts', 'users',
                         ['user_id'], ['id'])
```

### Issue 6: "No module named 'app.models'"

**Cause:** Python path not set correctly

**Solution:**

```ini
# alembic.ini
[alembic]
prepend_sys_path = .

# Or set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 7: Slow Migrations on Large Tables

**Solution:**

```python
def upgrade():
    # Create index CONCURRENTLY (PostgreSQL)
    op.execute('CREATE INDEX CONCURRENTLY ix_users_email ON users(email)')

    # Or add column without locking
    op.execute('ALTER TABLE users ADD COLUMN status VARCHAR(20) DEFAULT NULL')
```

## 📝 Exercises

### Exercise 1: Create Migration for Comments

Create a migration for a comments table with:

- id, content, post_id, user_id
- Timestamps
- Foreign keys to posts and users

### Exercise 2: Add Soft Deletes

Create a migration to add soft delete support:

- Add `deleted_at` column to users and posts
- Create index on deleted_at

### Exercise 3: Seed Data

Create a comprehensive seed script that:

- Creates 100 users
- Creates 500 posts with random categories
- Creates 1000 comments
- Uses Faker for realistic data

### Exercise 4: Safe Column Rename

Create a migration that safely renames a column with data preservation:

- Rename `users.name` to `users.full_name`
- Ensure data is preserved
- Include proper downgrade logic

### Exercise 5: Add Enum Type

Create a migration that:

- Adds a PostgreSQL ENUM type for user status ('active', 'inactive', 'suspended')
- Adds a `status` column using this enum
- Includes proper downgrade that removes both column and enum type

## 📋 Quick Reference

### Common Commands Cheat Sheet

```bash
# Setup
alembic init alembic                                    # Initialize Alembic
alembic revision --autogenerate -m "message"            # Create migration

# Running Migrations
alembic upgrade head                                     # Run all pending
alembic upgrade +1                                       # Run one migration
alembic upgrade abc123                                   # Upgrade to specific revision

# Rolling Back
alembic downgrade -1                                     # Rollback one
alembic downgrade base                                   # Rollback all
alembic downgrade abc123                                 # Downgrade to specific revision

# Information
alembic current                                          # Show current version
alembic history                                          # Show all migrations
alembic history --verbose                                # Detailed history
alembic heads                                            # Show pending migrations
alembic show abc123                                      # Show specific migration

# Maintenance
alembic stamp head                                       # Mark database as current
alembic merge heads -m "merge"                          # Merge branches
```

### Migration Operations Quick Reference

```python
# Tables
op.create_table('table_name', ...)
op.drop_table('table_name')
op.rename_table('old_name', 'new_name')

# Columns
op.add_column('table', sa.Column('name', sa.String(50)))
op.drop_column('table', 'column_name')
op.alter_column('table', 'column', new_column_name='new_name')

# Indexes
op.create_index('ix_name', 'table', ['column'])
op.drop_index('ix_name', table_name='table')

# Constraints
op.create_foreign_key('fk_name', 'source', 'target', ['col'], ['id'])
op.drop_constraint('constraint_name', 'table', type_='foreignkey')
op.create_unique_constraint('uq_name', 'table', ['column'])

# Raw SQL
op.execute("SQL STATEMENT")
op.execute(sa.text("SQL WITH :param").bindparams(param='value'))
```

## 🎓 Advanced Topics (Reference)

### CI/CD Integration

**GitHub Actions Example:**

```yaml
# .github/workflows/migrations.yml
name: Database Migrations

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-migrations:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run migrations
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
        run: |
          alembic upgrade head

      - name: Test rollback
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/testdb
        run: |
          alembic downgrade -1
          alembic upgrade head
```

**Production Deployment Script:**

```bash
#!/bin/bash
# deploy-migrations.sh

set -e  # Exit on error

echo "🔍 Checking migration status..."
alembic current

echo "📋 Showing pending migrations..."
alembic history --verbose

echo "⚠️  Creating backup..."
BACKUP_FILE="backup_$(date +%Y%m%d_%H%M%S).sql"
pg_dump $DATABASE_URL > $BACKUP_FILE
echo "✅ Backup created: $BACKUP_FILE"

echo "🚀 Running migrations..."
alembic upgrade head

echo "✅ Migrations completed successfully!"
alembic current
```

### Multiple Database Support

```python
# alembic/env.py
def run_migrations_online():
    # Primary database
    connectable = create_engine(PRIMARY_DB_URL)

    # Secondary database
    secondary_connectable = create_engine(SECONDARY_DB_URL)

    # Run migrations on both
```

### Branching Migrations

```bash
# Create a branch from a specific revision
alembic revision -m "branch" --head=abc123 --splice

# Merge multiple heads
alembic merge heads -m "merge feature branches"
```

### Migration Testing

```python
# tests/test_migrations.py
import pytest
from alembic import command
from alembic.config import Config

def test_migration_upgrade_downgrade():
    """Test that migrations can be applied and rolled back"""
    alembic_cfg = Config("alembic.ini")

    # Start from base
    command.downgrade(alembic_cfg, "base")

    # Upgrade to head
    command.upgrade(alembic_cfg, "head")

    # Downgrade one step
    command.downgrade(alembic_cfg, "-1")

    # Upgrade again
    command.upgrade(alembic_cfg, "head")

def test_migration_is_reversible():
    """Test each migration individually"""
    alembic_cfg = Config("alembic.ini")

    # Get all revisions
    from alembic.script import ScriptDirectory
    script = ScriptDirectory.from_config(alembic_cfg)

    for revision in script.walk_revisions():
        # Test upgrade
        command.upgrade(alembic_cfg, revision.revision)

        # Test downgrade
        if revision.down_revision:
            command.downgrade(alembic_cfg, revision.down_revision)
```

### Zero-Downtime Migrations

For production systems that can't afford downtime:

```python
# Migration 1: Add column as nullable
def upgrade():
    op.add_column('users', sa.Column('status', sa.String(20), nullable=True))

# Deploy new code that writes to both old and new schema

# Migration 2: Backfill data
def upgrade():
    op.execute("UPDATE users SET status = 'active' WHERE status IS NULL")

# Migration 3: Make column non-nullable
def upgrade():
    op.alter_column('users', 'status', nullable=False)

# Remove old code that writes to old schema
```

## 💻 Code Examples

### Standalone Application

📁 [`code-examples/chapter-07/standalone/`](code-examples/chapter-07/standalone/)

An **E-commerce Catalog API** demonstrating:

- Alembic migrations
- Database seeding with Faker
- Factory patterns for test data
- Migration version control

**Run it:**

```bash
cd code-examples/chapter-07/standalone
pip install -r requirements.txt
alembic upgrade head
python seed.py
uvicorn ecommerce_catalog:app --reload
```

### Progressive Application

📁 [`code-examples/chapter-07/progressive/`](code-examples/chapter-07/progressive/)

**Task Manager v7** - Adds migrations to v6:

- Alembic configuration
- Migration scripts
- Database seeders
- Version control for schema

### Code Snippets

📁 [`code-examples/chapter-07/snippets/`](code-examples/chapter-07/snippets/)

- **`migration_example.py`** - Migration patterns and examples
- **`database_seeder.py`** - Database seeding with factories

### Comprehensive Application

See **[TaskForce Pro](code-examples/comprehensive-app/)**.

## 🔗 Next Steps

**Next Chapter:** [Chapter 08: File Storage & Management](08-file-storage.md)

Learn how to handle file uploads, cloud storage, and image processing.

## 📚 Further Reading

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Alembic Cookbook](https://alembic.sqlalchemy.org/en/latest/cookbook.html)
- [Faker Documentation](https://faker.readthedocs.io/)
