"""
Chapter 07: Migrations & Seeders - Task Manager v7 with Alembic

Progressive Build: Adds Alembic migrations
- Database migrations with Alembic
- Seeder functions
- Version control for schema
- Migration workflow

Previous: chapter-06/progressive (database)
Next: chapter-08/progressive (cloud storage)

Setup:
1. pip install -r requirements.txt
2. alembic init alembic
3. alembic revision --autogenerate -m "Initial schema"
4. alembic upgrade head
5. python seed.py (to seed data)
6. uvicorn task_manager_v7_migrations:app --reload

Note: Same code as v6, but now with migration support.
See alembic/ directory and seed.py for migration examples.
"""

# Same models as v6 - migrations manage schema changes
from task_manager_v6_database import *

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     TASK MANAGER API V7 - Chapter 07                     ║
    ╚══════════════════════════════════════════════════════════╝
    
    Progressive Build:
    ✓ Chapter 07: Migrations (Alembic) ← You are here
    
    Migration Commands:
    - alembic revision --autogenerate -m "message"
    - alembic upgrade head
    - alembic downgrade -1
    - python seed.py (seed data)
    """)
    uvicorn.run("task_manager_v7_migrations:app", host="0.0.0.0", port=8000, reload=True)

