"""
CONCEPT: Database Seeder
- Populates database with test data
- Like Laravel's DatabaseSeeder
"""

from sqlalchemy.orm import Session
from task_manager_v6_database import SessionLocal, User, Task
from datetime import datetime, timedelta

def seed_database():
    """Seed the database with sample data."""
    db = SessionLocal()
    
    try:
        # Clear existing data
        db.query(Task).delete()
        db.query(User).delete()
        db.commit()
        
        # Create users
        users = [
            User(username="alice", email="alice@example.com", password_hash="password123"),
            User(username="bob", email="bob@example.com", password_hash="password123"),
        ]
        
        for user in users:
            db.add(user)
        db.commit()
        
        # Refresh to get IDs
        for user in users:
            db.refresh(user)
        
        # Create tasks for Alice
        alice_tasks = [
            Task(
                title="Complete FastAPI tutorial",
                priority="high",
                completed=False,
                due_date=(datetime.now() + timedelta(days=2)).isoformat(),
                user_id=users[0].id
            ),
            Task(
                title="Write documentation",
                priority="medium",
                completed=True,
                user_id=users[0].id
            ),
            Task(
                title="Review pull requests",
                priority="low",
                completed=False,
                user_id=users[0].id
            ),
        ]
        
        # Create tasks for Bob
        bob_tasks = [
            Task(
                title="Setup CI/CD pipeline",
                priority="high",
                completed=False,
                user_id=users[1].id
            ),
            Task(
                title="Deploy to production",
                priority="high",
                completed=False,
                user_id=users[1].id
            ),
        ]
        
        for task in alice_tasks + bob_tasks:
            db.add(task)
        
        db.commit()
        
        print(f"✅ Seeded database with {len(users)} users and {len(alice_tasks + bob_tasks)} tasks")
        
    finally:
        db.close()

if __name__ == "__main__":
    print("🌱 Seeding database...")
    seed_database()
    print("✓ Done!")

