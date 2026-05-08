"""SQLAlchemy models for HIL approvals and investigations."""

from datetime import datetime

from sqlalchemy import Column, DateTime, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class HilApproval(Base):
    """Pending or resolved HIL approvals for investigation actions."""

    __tablename__ = "hil_approvals"

    investigation_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    model_uri = Column(String, nullable=False, index=True)
    recommended_action = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)  # pending/approved/rejected/stale
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    superseded_by = Column(String, nullable=True)  # investigation_id of newer investigation


class Investigation(Base):
    """Investigation record for tracking open and resolved investigations."""

    __tablename__ = "investigations"

    investigation_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False, index=True)
    model_version = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # green/yellow/red or LOW/MEDIUM/HIGH
    status = Column(String, default="open", nullable=False)  # open/resolved/stale
    thread_id = Column(String, nullable=False)  # LangGraph thread ID for resumption
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
