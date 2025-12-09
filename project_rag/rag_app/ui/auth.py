#auth.py
from __future__ import annotations
import time
from typing import Tuple

import bcrypt
import streamlit as st

DEFAULT_HASH = "$2b$12$5OUdfM6VM8RFPa54Q2brqO/mxlsUjHw83TU0xtQpFTNAa6zmQDcrq"  # 12345


def _load_creds():
    creds = st.secrets.get("credentials", {})
    users = creds.get("usernames", {})
    if not users:
        users = {
            "cristian": {"name": "Cristian Ortega", "password": DEFAULT_HASH}
        }
    cookie_name = st.secrets.get("credentials", {}).get("cookie_name", "ragapp_auth")
    cookie_key = st.secrets.get("credentials", {}).get("cookie_key", "supersecretkey-change-me")
    cookie_days = int(st.secrets.get("credentials", {}).get("cookie_expiry_days", 7))
    return users, cookie_name, cookie_key, cookie_days


def _check_password(raw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(raw.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def login() -> Tuple[bool, str, str]:
    users, _, _, cookie_days = _load_creds()

    st.sidebar.subheader("Sign in")
    u = st.sidebar.text_input("Username", key="auth_user_widget")
    p = st.sidebar.text_input("Password", type="password", key="auth_pass_widget")
    go = st.sidebar.button("Login", key="auth_login_btn")

    # initialize session flags
    if "auth_ok" not in st.session_state:
        st.session_state["auth_ok"] = False
        st.session_state["auth_name"] = ""
        st.session_state["auth_username"] = ""
        st.session_state["auth_exp"] = 0.0

    if go and not st.session_state["auth_ok"]:
        if u in users and _check_password(p, users[u]["password"]):
            st.session_state["auth_ok"] = True
            st.session_state["auth_name"] = users[u]["name"]
            st.session_state["auth_username"] = u
            st.session_state["auth_exp"] = time.time() + cookie_days * 86400
        else:
            st.sidebar.error("Invalid username/password")

    # expiry check
    if st.session_state["auth_ok"] and time.time() > st.session_state["auth_exp"]:
        st.session_state["auth_ok"] = False

    if st.session_state["auth_ok"]:
        return True, st.session_state["auth_name"], st.session_state["auth_username"]
    return False, "", ""
