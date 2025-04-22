"""
Web Development Utilities
========================

A collection of classes and functions for web development tasks.
Includes API client implementation, authentication handling,
and common web operations.

Features:
    - RESTful API client
    - JWT authentication
    - Request handling
    - Response processing
"""

class APIClient:
    """
    A flexible REST API client with authentication support.
    
    Attributes:
        base_url (str): Base URL for all API endpoints
        api_key (str): Optional API key for authentication
        session (requests.Session): Persistent session for requests
        
    Methods:
        get: Perform GET requests
        post: Perform POST requests
        put: Perform PUT requests
        delete: Perform DELETE requests
    
    Example:
        >>> client = APIClient("https://api.example.com", api_key="secret")
        >>> response = client.get("/users")
        >>> print(response)
    """
    
import requests
from typing import Dict, Any
from datetime import datetime, timedelta
import jwt

class APIClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        """Execute GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        response = self.session.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        """Execute POST request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = self._get_headers()
        response = self.session.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    
    def _get_headers(self) -> Dict:
        """Generate request headers."""
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers

class JWTAuth:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, user_id: str, expiry_hours: int = 24) -> str:
        """Generate JWT token."""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=expiry_hours)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Dict:
        """Verify JWT token."""
        return jwt.decode(token, self.secret_key, algorithms=['HS256'])