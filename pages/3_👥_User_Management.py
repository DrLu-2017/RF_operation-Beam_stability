import streamlit as st
import yaml
from yaml.loader import SafeLoader
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(page_title="User Management", page_icon="👥", layout="wide")

if not st.session_state.get("authentication_status"):
    st.info("请先前往主页 (Home) 登录。 / Please login from the Home page.")
    st.stop()

is_admin = st.session_state.get("username") == "admin"
if not is_admin:
    st.error("权限被拒绝：此页面仅限系统管理员访问。 / Access Denied: Administrator level required.")
    st.stop()

try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("streamlit_authenticator not installed.")
    st.stop()

st.title("👥 User Management (Admin Dashboard)")
st.markdown("Here you can create new user accounts, reset passwords, or completely remove users from the system.")

# Load current config
config_path = project_root / 'auth_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

# Render Tabs
tab1, tab2, tab3 = st.tabs(["➕ Add New User", "🗑️ Delete User", "🔑 Admin Override Password"])

with tab1:
    st.subheader("Register a new user account")
    try:
        email, username, name = authenticator.register_user(location='main', captcha=False)
        if email:
            st.success(f"User '{username}' registered successfully!")
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
    except Exception as e:
        st.error(f"Registration Error: {e}")

with tab2:
    st.subheader("Delete an existing user")
    existing_users = [u for u in config['credentials']['usernames'].keys() if u != 'admin']
    if not existing_users:
        st.info("No other users available to delete.")
    else:
        target_user_del = st.selectbox("Select user to remove:", options=existing_users)
        st.warning(f"Warning: Deleting user '{target_user_del}' cannot be undone. Their SSH profiles will remain on disk but their login will be deleted.")
        if st.button(f"🗑️ Confirm Delete '{target_user_del}'", type="primary"):
            del config['credentials']['usernames'][target_user_del]
            with open(config_path, 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
            st.success(f"User '{target_user_del}' has been permanently deleted.")
            st.rerun()

with tab3:
    st.subheader("Admin Override Password")
    st.markdown("Use this to forcefully set a new password for any user without knowing their current password.")
    all_users = list(config['credentials']['usernames'].keys())
    target_user_pwd = st.selectbox("Select user:", options=all_users, key='pwd_user_select')
    new_password = st.text_input("Enter new password:", type="password")
    if st.button("🔑 Set New Password", type="primary"):
        if new_password:
            try:
                import bcrypt
                salt = bcrypt.gensalt()
                hashed = bcrypt.hashpw(new_password.encode('utf-8'), salt).decode('utf-8')
                config['credentials']['usernames'][target_user_pwd]['password'] = hashed
                with open(config_path, 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                st.success(f"Password for '{target_user_pwd}' updated successfully!")
            except Exception as e:
                st.error(f"Error updating password: {e}")
        else:
            st.error("Password cannot be empty.")

st.markdown("---")
st.subheader("📋 Current User Roster")
user_table = []
for u, details in config['credentials']['usernames'].items():
    user_table.append({
        "Username": u,
        "Name": details.get("name", ""),
        "Email": details.get("email", ""),
        "Role": "Administrator" if u == "admin" else "Lab Member"
    })
st.dataframe(user_table, use_container_width=True)
