"""SQLAlchemy 2.0 declarative base for the platform service."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Single Base for all platform ORM models. Imported by Alembic env."""
