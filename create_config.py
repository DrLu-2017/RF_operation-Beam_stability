import yaml
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

data = {
    'credentials': {
        'usernames': {
            'user1': {
                'email': 'user1@lab.com',
                'name': 'Lab User 1',
                'password': hash_password('labuser'),
                'logged_in': False
            },
            'admin': {
                'email': 'admin@lab.com',
                'name': 'Admin',
                'password': hash_password('admin123'),
                'logged_in': False
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_signature_key',
        'name': 'mbtrack2_remote_login'
    },
    'pre-authorized': {
        'emails': []
    }
}

with open('auth_config.yaml', 'w') as f:
    yaml.dump(data, f)
