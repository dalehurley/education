"""
Chapter 07 Snippet: Database Seeders

Patterns for seeding test data.
Compare to Laravel's seeders and factories.
"""

from faker import Faker
from sqlalchemy.orm import Session
import random

fake = Faker()


# CONCEPT: Factory Pattern
class UserFactory:
    """
    Factory for generating user data.
    Like Laravel's factories.
    """
    
    @staticmethod
    def make(count: int = 1) -> list:
        """Generate user data without saving."""
        users = []
        for _ in range(count):
            users.append({
                "username": fake.user_name(),
                "email": fake.email(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "is_active": random.choice([True, False])
            })
        return users
    
    @staticmethod
    def create(db: Session, count: int = 1):
        """Generate and save users."""
        from sqlalchemy_models import User
        
        users = []
        for _ in range(count):
            user = User(
                username=fake.user_name(),
                email=fake.email(),
                is_active=True
            )
            db.add(user)
            users.append(user)
        
        db.commit()
        return users


class PostFactory:
    """Factory for blog posts."""
    
    @staticmethod
    def create(db: Session, author_id: int, count: int = 1):
        from sqlalchemy_models import Post
        
        posts = []
        for _ in range(count):
            post = Post(
                title=fake.sentence(),
                content=fake.text(500),
                published=random.choice([True, False]),
                author_id=author_id
            )
            db.add(post)
            posts.append(post)
        
        db.commit()
        return posts


# CONCEPT: Seeder Class
class DatabaseSeeder:
    """
    Main seeder class.
    Like Laravel's DatabaseSeeder.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def seed_users(self, count: int = 10):
        """Seed users."""
        print(f"Seeding {count} users...")
        users = UserFactory.create(self.db, count)
        print(f"✓ Created {len(users)} users")
        return users
    
    def seed_posts(self, users: list, posts_per_user: int = 5):
        """Seed posts for users."""
        print(f"Seeding posts...")
        total = 0
        
        for user in users:
            posts = PostFactory.create(self.db, user.id, posts_per_user)
            total += len(posts)
        
        print(f"✓ Created {total} posts")
    
    def seed_all(self):
        """
        Seed entire database.
        Like Laravel's DatabaseSeeder::run()
        """
        print("\n" + "=" * 50)
        print("  DATABASE SEEDING")
        print("=" * 50 + "\n")
        
        # Clear existing data
        print("Clearing existing data...")
        from sqlalchemy_models import User, Post
        self.db.query(Post).delete()
        self.db.query(User).delete()
        self.db.commit()
        
        # Seed users
        users = self.seed_users(50)
        
        # Seed posts
        self.seed_posts(users, posts_per_user=5)
        
        print("\n" + "=" * 50)
        print("  SEEDING COMPLETE!")
        print("=" * 50 + "\n")


# Usage example
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy_models import Base
    
    # Setup
    engine = create_engine("sqlite:///example.db")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    
    # Run seeder
    seeder = DatabaseSeeder(db)
    seeder.seed_all()
    
    db.close()

