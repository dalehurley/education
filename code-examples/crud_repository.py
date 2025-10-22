"""
Generic CRUD Repository Pattern
Referenced in Chapter 06: Database with SQLAlchemy
"""

from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import DeclarativeBase

ModelType = TypeVar("ModelType", bound=DeclarativeBase)

class CRUDRepository(Generic[ModelType]):
    """Generic CRUD repository for database operations"""
    
    def __init__(self, model: Type[ModelType]):
        self.model = model
    
    async def get(
        self,
        db: AsyncSession,
        id: int
    ) -> Optional[ModelType]:
        """Get single record by ID"""
        result = await db.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()
    
    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Get multiple records with pagination"""
        result = await db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return result.scalars().all()
    
    async def create(
        self,
        db: AsyncSession,
        obj_in: dict
    ) -> ModelType:
        """Create new record"""
        db_obj = self.model(**obj_in)
        db.add(db_obj)
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def update(
        self,
        db: AsyncSession,
        id: int,
        obj_in: dict
    ) -> Optional[ModelType]:
        """Update existing record"""
        db_obj = await self.get(db, id)
        if not db_obj:
            return None
        
        for key, value in obj_in.items():
            setattr(db_obj, key, value)
        
        await db.commit()
        await db.refresh(db_obj)
        return db_obj
    
    async def delete(
        self,
        db: AsyncSession,
        id: int
    ) -> bool:
        """Delete record"""
        db_obj = await self.get(db, id)
        if not db_obj:
            return False
        
        await db.delete(db_obj)
        await db.commit()
        return True

# Usage example
"""
from app.models.user import User

user_repository = CRUDRepository(User)

# In your endpoint
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await user_repository.get(db, user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user

@app.post("/users")
async def create_user(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    user_dict = user_data.model_dump()
    user = await user_repository.create(db, user_dict)
    return user
"""

