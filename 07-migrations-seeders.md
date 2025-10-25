# Chapter 07: Migrations & Seeders

## 🎯 Learning Objectives

By the end of this chapter, you will:

- Set up Alembic for database migrations
- Create and run migrations
- Handle migration rollbacks
- Seed your database with test data
- Manage database versions

## 🔄 Laravel vs Alembic

| Feature          | Laravel                        | Alembic                    |
| ---------------- | ------------------------------ | -------------------------- |
| Tool             | Built-in migrations            | Alembic (separate package) |
| Create migration | `php artisan make:migration`   | `alembic revision`         |
| Run migrations   | `php artisan migrate`          | `alembic upgrade head`     |
| Rollback         | `php artisan migrate:rollback` | `alembic downgrade -1`     |
| Status           | `php artisan migrate:status`   | `alembic current`          |
| Seed             | `php artisan db:seed`          | Custom scripts             |
| Fresh            | `php artisan migrate:fresh`    | Drop + upgrade             |

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

## 🎓 Advanced Topics (Reference)

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
alembic revision -m "branch" --head=abc123 --splice
```

### Migration Testing

```python
# tests/test_migrations.py
def test_migration_upgrade_downgrade():
    alembic_cfg = Config("alembic.ini")

    # Upgrade
    command.upgrade(alembic_cfg, "head")

    # Downgrade
    command.downgrade(alembic_cfg, "-1")

    # Upgrade again
    command.upgrade(alembic_cfg, "head")
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
