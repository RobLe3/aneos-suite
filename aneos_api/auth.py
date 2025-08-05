"""
Authentication and authorization for aNEOS API.

Provides API key authentication, user management, and role-based access control.
"""

from typing import Optional, Dict, Any
import logging
import hashlib
import secrets
from datetime import datetime, timedelta

try:
    from fastapi import HTTPException, Depends, Security
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.security.api_key import APIKeyHeader
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logging.warning("FastAPI not available, authentication disabled")

try:
    from .models import UserRole
except ImportError:
    # Fallback enum for UserRole
    from enum import Enum
    class UserRole(str, Enum):
        ADMIN = "admin"
        ANALYST = "analyst"
        VIEWER = "viewer"

logger = logging.getLogger(__name__)

# Mock user database (in production would use actual database)
MOCK_USERS = {
    'admin': {
        'user_id': 'admin_001',
        'username': 'admin',
        'email': 'admin@aneos.local',
        'password_hash': 'mock_admin_hash',
        'role': UserRole.ADMIN,
        'api_keys': ['aneos_admin_key_123'],
        'created_at': datetime.now(),
        'last_login': datetime.now(),
        'is_active': True
    },
    'analyst': {
        'user_id': 'analyst_001',
        'username': 'analyst',
        'email': 'analyst@aneos.local',
        'password_hash': 'mock_analyst_hash',
        'role': UserRole.ANALYST,
        'api_keys': ['aneos_analyst_key_456'],
        'created_at': datetime.now(),
        'last_login': datetime.now(),
        'is_active': True
    },
    'viewer': {
        'user_id': 'viewer_001',
        'username': 'viewer',
        'email': 'viewer@aneos.local',
        'password_hash': 'mock_viewer_hash',
        'role': UserRole.VIEWER,
        'api_keys': ['aneos_viewer_key_789'],
        'created_at': datetime.now(),
        'last_login': datetime.now(),
        'is_active': True
    }
}

# API key to user mapping
API_KEY_MAP = {}
for user_data in MOCK_USERS.values():
    for api_key in user_data['api_keys']:
        API_KEY_MAP[api_key] = user_data

if HAS_FASTAPI:
    # Security schemes
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    bearer_token = HTTPBearer(auto_error=False)
else:
    # Mock security schemes
    api_key_header = None
    bearer_token = None

class AuthManager:
    """Manages authentication and authorization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session_timeout = timedelta(hours=24)
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def authenticate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by API key."""
        if not api_key:
            return None
            
        user_data = API_KEY_MAP.get(api_key)
        if user_data and user_data['is_active']:
            # Update last login
            user_data['last_login'] = datetime.now()
            return user_data
        
        return None
    
    def authenticate_bearer_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Authenticate user by bearer token."""
        # Mock token authentication (would validate JWT in production)
        if token == "mock_admin_token":
            return MOCK_USERS['admin']
        elif token == "mock_analyst_token":
            return MOCK_USERS['analyst']
        elif token == "mock_viewer_token":
            return MOCK_USERS['viewer']
        
        return None
    
    def create_api_key(self, user_id: str) -> str:
        """Create a new API key for a user."""
        api_key = f"aneos_{secrets.token_urlsafe(32)}"
        
        # Add to user's API keys (mock implementation)
        for user_data in MOCK_USERS.values():
            if user_data['user_id'] == user_id:
                user_data['api_keys'].append(api_key)
                API_KEY_MAP[api_key] = user_data
                break
        
        return api_key
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in API_KEY_MAP:
            user_data = API_KEY_MAP[api_key]
            user_data['api_keys'].remove(api_key)
            del API_KEY_MAP[api_key]
            return True
        
        return False
    
    def check_permission(self, user: Dict[str, Any], required_role: UserRole) -> bool:
        """Check if user has required role/permission."""
        user_role = user.get('role')
        
        if user_role == UserRole.ADMIN:
            return True
        elif user_role == UserRole.ANALYST and required_role in [UserRole.ANALYST, UserRole.VIEWER]:
            return True
        elif user_role == UserRole.VIEWER and required_role == UserRole.VIEWER:
            return True
        
        return False

class APIKeyAuth:
    """API Key authentication utility."""
    
    @staticmethod
    def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user data."""
        return API_KEY_MAP.get(api_key) if api_key else None

# Dependency functions for FastAPI
async def get_current_user(
    api_key: Optional[str] = Security(api_key_header) if HAS_FASTAPI else None,
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_token) if HAS_FASTAPI else None
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from API key or bearer token."""
    if not HAS_FASTAPI:
        return None
    
    # Try API key authentication first
    if api_key:
        user = API_KEY_MAP.get(api_key)
        if user and user['is_active']:
            return user
    
    # Try bearer token authentication
    if bearer_token:
        auth_manager = AuthManager({})
        user = auth_manager.authenticate_bearer_token(bearer_token.credentials)
        if user:
            return user
    
    # No authentication provided - this is optional for most endpoints
    return None

async def require_authentication(
    current_user: Optional[Dict] = Depends(get_current_user) if HAS_FASTAPI else None
) -> Dict[str, Any]:
    """Require authentication - raises 401 if not authenticated."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user

async def require_role(required_role: UserRole):
    """Create a dependency that requires a specific role."""
    async def role_checker(current_user: Dict = Depends(require_authentication) if HAS_FASTAPI else None) -> Dict[str, Any]:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        auth_manager = AuthManager({})
        if not auth_manager.check_permission(current_user, required_role):
            raise HTTPException(
                status_code=403, 
                detail=f"Role {required_role.value} or higher required"
            )
        
        return current_user
    
    return role_checker

async def require_admin(current_user: Dict = Depends(require_authentication) if HAS_FASTAPI else None) -> Dict[str, Any]:
    """Require admin role."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if current_user.get('role') != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")
    
    return current_user

async def require_analyst_or_admin(current_user: Dict = Depends(require_authentication) if HAS_FASTAPI else None) -> Dict[str, Any]:
    """Require analyst or admin role."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    user_role = current_user.get('role')
    if user_role not in [UserRole.ADMIN, UserRole.ANALYST]:
        raise HTTPException(status_code=403, detail="Analyst or admin role required")
    
    return current_user

# Utility functions
def hash_password(password: str) -> str:
    """Hash a password using SHA-256 (use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return hash_password(password) == password_hash

def generate_session_token() -> str:
    """Generate a secure session token."""
    return secrets.token_urlsafe(32)

def is_api_key_valid(api_key: str) -> bool:
    """Check if API key is valid and active."""
    user_data = API_KEY_MAP.get(api_key)
    return user_data is not None and user_data['is_active']

def get_user_permissions(user: Dict[str, Any]) -> list:
    """Get list of permissions for a user based on their role."""
    role = user.get('role')
    
    if role == UserRole.ADMIN:
        return [
            'read_analysis', 'write_analysis', 'read_prediction', 'write_prediction',
            'read_monitoring', 'write_monitoring', 'read_admin', 'write_admin',
            'manage_users', 'manage_system', 'manage_training'
        ]
    elif role == UserRole.ANALYST:
        return [
            'read_analysis', 'write_analysis', 'read_prediction', 'write_prediction',
            'read_monitoring', 'start_training'
        ]
    elif role == UserRole.VIEWER:
        return [
            'read_analysis', 'read_prediction', 'read_monitoring'
        ]
    
    return []

def create_mock_user(username: str, email: str, role: UserRole, password: str) -> Dict[str, Any]:
    """Create a mock user (for testing purposes)."""
    user_id = f"{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    api_key = f"aneos_{secrets.token_urlsafe(16)}"
    
    user_data = {
        'user_id': user_id,
        'username': username,
        'email': email,
        'password_hash': hash_password(password),
        'role': role,
        'api_keys': [api_key],
        'created_at': datetime.now(),
        'last_login': None,
        'is_active': True
    }
    
    # Add to mock database
    MOCK_USERS[username] = user_data
    API_KEY_MAP[api_key] = user_data
    
    return user_data

# Initialize with some test API keys for development
logger.info("Authentication system initialized with mock users")
logger.info("Available API keys for testing:")
for username, user_data in MOCK_USERS.items():
    for api_key in user_data['api_keys']:
        logger.info(f"  {username} ({user_data['role'].value}): {api_key}")