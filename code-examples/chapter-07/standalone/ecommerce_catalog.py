"""
Chapter 07: Migrations & Seeders - E-commerce Product Catalog

Demonstrates:
- Alembic migrations setup
- Database seeding
- Factory patterns
- Migration version control

Setup:
1. alembic init alembic
2. Configure alembic.ini
3. alembic revision --autogenerate -m "initial"
4. alembic upgrade head
5. python ecommerce_catalog.py seed
6. uvicorn ecommerce_catalog:app --reload
"""

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, func
from pydantic import BaseModel
from typing import List
from datetime import datetime
import random

DATABASE_URL = "sqlite+aiosqlite:///./ecommerce.db"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# Models
class Category(Base):
    __tablename__ = "categories"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    products = relationship("Product", back_populates="category")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(String(1000))
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)
    category_id = Column(Integer, ForeignKey("categories.id"))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    category = relationship("Category", back_populates="products")

# Schemas
class CategoryResponse(BaseModel):
    id: int
    name: str
    slug: str
    class Config:
        from_attributes = True

class ProductResponse(BaseModel):
    id: int
    name: str
    price: float
    stock: int
    category: CategoryResponse
    class Config:
        from_attributes = True

# Database dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Create tables
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# Seeder functions
async def seed_categories(db: AsyncSession):
    """CONCEPT: Database Seeding - Like Laravel seeders"""
    categories = [
        {"name": "Electronics", "slug": "electronics"},
        {"name": "Clothing", "slug": "clothing"},
        {"name": "Books", "slug": "books"},
        {"name": "Home & Garden", "slug": "home-garden"},
    ]
    for cat_data in categories:
        category = Category(**cat_data)
        db.add(category)
    await db.commit()

async def seed_products(db: AsyncSession):
    """CONCEPT: Factory Pattern for test data"""
    from sqlalchemy import select
    result = await db.execute(select(Category))
    categories = result.scalars().all()
    
    product_names = [
        "Laptop", "Mouse", "Keyboard", "Monitor", "Headphones",
        "T-Shirt", "Jeans", "Jacket", "Shoes", "Hat",
        "Novel", "Cookbook", "Dictionary", "Atlas", "Magazine",
        "Chair", "Table", "Lamp", "Vase", "Clock"
    ]
    
    for name in product_names:
        product = Product(
            name=name,
            description=f"High quality {name.lower()}",
            price=round(random.uniform(10, 500), 2),
            stock=random.randint(0, 100),
            category_id=random.choice(categories).id
        )
        db.add(product)
    await db.commit()

app = FastAPI(title="E-commerce Catalog - Chapter 07")

@app.on_event("startup")
async def startup():
    await create_tables()

@app.get("/")
async def root():
    return {"message": "E-commerce Catalog API", "docs": "/docs"}

@app.get("/products", response_model=List[ProductResponse])
async def list_products(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload
    result = await db.execute(select(Product).options(selectinload(Product.category)))
    return result.scalars().all()

@app.get("/categories", response_model=List[CategoryResponse])
async def list_categories(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Category))
    return result.scalars().all()

if __name__ == "__main__":
    import sys
    import asyncio
    
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        async def run_seeders():
            await create_tables()
            async with AsyncSessionLocal() as db:
                await seed_categories(db)
                await seed_products(db)
            print("✓ Database seeded successfully!")
        
        asyncio.run(run_seeders())
    else:
        import uvicorn
        uvicorn.run("ecommerce_catalog:app", host="0.0.0.0", port=8000, reload=True)

