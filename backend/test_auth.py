from fastapi.testclient import TestClient
import unittest
import string
import random
from app.main import app

client = TestClient(app)

def get_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

class TestAuth(unittest.TestCase):
    
    def setUp(self):
        # Create unique user for each test
        self.email = f"test_{get_random_string()}@example.com"
        self.password = "testpassword123"
        
    def test_signup(self):
        print(f"\nTesting Signup with {self.email}...")
        response = client.post("/auth/signup", json={
            "email": self.email,
            "password": self.password
        })
        print(f"Signup Response: {response.status_code}")
        if response.status_code != 200:
            print(response.json())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["email"], self.email)
        self.assertTrue("id" in data)
        return data

    def test_login(self):
        # Signup first
        self.test_signup()
        
        print(f"\nTesting Login...")
        response = client.post("/auth/login", data={
            "username": self.email, # OAuth2 form uses 'username' field
            "password": self.password
        })
        print(f"Login Response: {response.status_code}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue("access_token" in data)
        self.assertEqual(data["token_type"], "bearer")
        
        # Test protected endpoint
        token = data["access_token"]
        print(f"\nTesting Protected Endpoint (/auth/me)...")
        headers = {"Authorization": f"Bearer {token}"}
        me_response = client.get("/auth/me", headers=headers)
        print(f"Me Response: {me_response.status_code}")
        self.assertEqual(me_response.status_code, 200)
        me_data = me_response.json()
        self.assertEqual(me_data["email"], self.email)

if __name__ == "__main__":
    unittest.main()
