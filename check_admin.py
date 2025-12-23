import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.models.user import User

def check_admin_user(email):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user:
            print(f"User: {user.email}, Role: {user.role}, Role Type: {type(user.role)}")
            if str(user.role) == "ADMIN" or user.role == "ADMIN" or getattr(user.role, "value", None) == "ADMIN":
                print("User is ADMIN")
            else:
                print("User is NOT ADMIN")
        else:
            print("User not found")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        email = sys.argv[1]
        check_admin_user(email)
    else:
        print("Usage: python check_admin.py <email>")

