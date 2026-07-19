"""Session auth for the PDF viewer: open sign-up, login, logout.

Hand-rolled on Flask sessions + werkzeug password hashing (no extra
dependency). The app's before_request hook enforces login on everything
except the endpoints in PUBLIC_ENDPOINTS.
"""

import logging
import re

from flask import (Blueprint, jsonify, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import db

log = logging.getLogger(__name__)

bp = Blueprint("auth", __name__)

PUBLIC_ENDPOINTS = {"auth.login", "auth.login_post",
                    "auth.signup", "auth.signup_post", "static"}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD = 8


def current_uid() -> int | None:
    return session.get("uid")


def require_login():
    """before_request hook: 401 for API calls, redirect for page loads."""
    if request.endpoint in PUBLIC_ENDPOINTS or current_uid() is not None:
        return None
    wants_json = (request.path.startswith(("/api/", "/logs/", "/upload"))
                  or request.method != "GET")
    if wants_json:
        return jsonify({"error": "not logged in"}), 401
    return redirect(url_for("auth.login"))


def _login_page(mode: str, error: str | None = None):
    return render_template("login.html", mode=mode, error=error,
                           username=request.form.get("username", ""))


@bp.get("/login")
def login():
    if current_uid() is not None:
        return redirect("/")
    return _login_page("login")


@bp.post("/login")
def login_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    user = db.get_user_by_name(username) if username else None
    if user is None or not check_password_hash(user["password_hash"], password):
        return _login_page("login", "Invalid username or password."), 401
    session.clear()
    session["uid"] = user["id"]
    session["username"] = user["username"]
    session.permanent = True
    log.info(f"Login: {user['username']} (id {user['id']})")
    return redirect("/")


@bp.get("/signup")
def signup():
    if current_uid() is not None:
        return redirect("/")
    return _login_page("signup")


@bp.post("/signup")
def signup_post():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    if not USERNAME_RE.fullmatch(username):
        return _login_page(
            "signup", "Username must be 3-32 characters: letters, digits, _ . -"
        ), 400
    if len(password) < MIN_PASSWORD:
        return _login_page(
            "signup", f"Password must be at least {MIN_PASSWORD} characters."
        ), 400
    user_id = db.create_user(username, generate_password_hash(password))
    if user_id is None:
        return _login_page("signup", "That username is already taken."), 409
    session.clear()
    session["uid"] = user_id
    session["username"] = username
    session.permanent = True
    log.info(f"Signup: {username} (id {user_id})")
    return redirect("/")


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
