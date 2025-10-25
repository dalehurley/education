"""
Chapter 01 Snippet: Data Processing with List Comprehensions

Demonstrates:
- List comprehensions
- Dictionary comprehensions
- Lambda functions
- filter() and map()
"""

from typing import List, Dict


# Sample data
users = [
    {"name": "Alice", "age": 30, "email": "alice@example.com"},
    {"name": "Bob", "age": 25, "email": "bob@example.com"},
    {"name": "Charlie", "age": 35, "email": "charlie@example.com"},
    {"name": "Diana", "age": 28, "email": "diana@example.com"},
    {"name": "Eve", "age": 22, "email": "eve@example.com"},
]


def get_emails(users: List[Dict]) -> List[str]:
    """
    Extract all user emails using list comprehension.
    
    Laravel equivalent: array_column($users, 'email')
    """
    return [user["email"] for user in users]


def get_adults_over_25(users: List[Dict]) -> List[Dict]:
    """
    Filter users who are over 25 years old.
    
    Laravel equivalent: 
    array_filter($users, fn($user) => $user['age'] > 25)
    """
    return [user for user in users if user["age"] > 25]


def create_age_map(users: List[Dict]) -> Dict[str, int]:
    """
    Create a dictionary mapping names to ages.
    
    Laravel equivalent: 
    array_combine(
        array_column($users, 'name'),
        array_column($users, 'age')
    )
    """
    return {user["name"]: user["age"] for user in users}


def get_user_summaries(users: List[Dict]) -> List[str]:
    """
    Create summary strings for each user.
    
    Demonstrates: Combining list comprehension with f-strings
    """
    return [
        f"{user['name']} ({user['age']}) - {user['email']}"
        for user in users
    ]


def get_senior_emails(users: List[Dict]) -> List[str]:
    """
    Get emails of users over 30, using chained comprehensions.
    
    Demonstrates: Complex filtering with comprehensions
    """
    return [
        user["email"] 
        for user in users 
        if user["age"] > 30
    ]


def transform_users(users: List[Dict]) -> List[Dict]:
    """
    Transform user data with computed fields.
    
    Demonstrates: Dictionary comprehension with transformations
    """
    return [
        {
            **user,  # Spread operator (like PHP's ...$user)
            "username": user["email"].split("@")[0],
            "age_group": "senior" if user["age"] > 30 else "junior"
        }
        for user in users
    ]


def main():
    """Demo all data processing functions."""
    print("Data Processing Examples")
    print("=" * 60)
    
    print("\n1. All Emails:")
    emails = get_emails(users)
    for email in emails:
        print(f"   - {email}")
    
    print("\n2. Users Over 25:")
    adults = get_adults_over_25(users)
    for user in adults:
        print(f"   - {user['name']} ({user['age']})")
    
    print("\n3. Age Map:")
    age_map = create_age_map(users)
    for name, age in age_map.items():
        print(f"   - {name}: {age}")
    
    print("\n4. User Summaries:")
    summaries = get_user_summaries(users)
    for summary in summaries:
        print(f"   - {summary}")
    
    print("\n5. Senior Emails (>30):")
    senior_emails = get_senior_emails(users)
    for email in senior_emails:
        print(f"   - {email}")
    
    print("\n6. Transformed Users:")
    transformed = transform_users(users)
    for user in transformed:
        print(f"   - {user['name']}: {user['username']} ({user['age_group']})")


if __name__ == "__main__":
    main()

