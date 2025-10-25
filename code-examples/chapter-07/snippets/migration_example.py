"""
Chapter 07 Snippet: Alembic Migration Examples

Example migration scripts and patterns.
Compare to Laravel's migrations.
"""

# CONCEPT: Migration Template
"""
This shows what an Alembic migration file looks like.
Generated with: alembic revision --autogenerate -m "message"

Like Laravel's: php artisan make:migration
"""

# Example Migration File Structure
MIGRATION_TEMPLATE = '''
"""Add users table

Revision ID: abc123
Create Date: 2025-01-24

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """
    CONCEPT: Upgrade (Apply Changes)
    Like Laravel's up() method
    """
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(50), nullable=False, unique=True),
        sa.Column('email', sa.String(100), nullable=False, unique=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    
    # Add index
    op.create_index('ix_users_email', 'users', ['email'])

def downgrade():
    """
    CONCEPT: Downgrade (Rollback Changes)
    Like Laravel's down() method
    """
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
'''

# CONCEPT: Add Column Migration
ADD_COLUMN_EXAMPLE = '''
def upgrade():
    # Add new column
    op.add_column('users',
        sa.Column('is_active', sa.Boolean(), server_default='1')
    )

def downgrade():
    op.drop_column('users', 'is_active')
'''

# CONCEPT: Modify Column Migration
MODIFY_COLUMN_EXAMPLE = '''
def upgrade():
    # Modify column type
    op.alter_column('users', 'username',
        existing_type=sa.String(50),
        type_=sa.String(100),
        existing_nullable=False
    )

def downgrade():
    op.alter_column('users', 'username',
        existing_type=sa.String(100),
        type_=sa.String(50),
        existing_nullable=False
    )
'''

# CONCEPT: Data Migration
DATA_MIGRATION_EXAMPLE = '''
def upgrade():
    # Insert data
    from sqlalchemy import table, column
    from sqlalchemy.sql import select
    
    users_table = table('users',
        column('username', sa.String),
        column('email', sa.String)
    )
    
    op.bulk_insert(users_table, [
        {'username': 'admin', 'email': 'admin@example.com'},
        {'username': 'user', 'email': 'user@example.com'}
    ])

def downgrade():
    op.execute("DELETE FROM users WHERE username IN ('admin', 'user')")
'''

if __name__ == "__main__":
    print("Migration Examples")
    print("=" * 50)
    print("\n1. Create Table Migration:")
    print(MIGRATION_TEMPLATE)
    print("\n2. Add Column:")
    print(ADD_COLUMN_EXAMPLE)
    print("\n3. Modify Column:")
    print(MODIFY_COLUMN_EXAMPLE)
    print("\n4. Data Migration:")
    print(DATA_MIGRATION_EXAMPLE)

