"""SQLAlchemy models for KRAKEN database schema."""
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class KrakenConversation(Base):
    """Model for kraken_conversations table."""

    __tablename__ = "kraken_conversations"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(String, nullable=False, index=True)
    model = Column(String, nullable=False)
    system_prompt_hash = Column(String(16), nullable=False)
    agent_version = Column(String, nullable=False)
    status = Column(String, default="active", nullable=False)
    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    ended_at = Column(DateTime, nullable=True)
    total_turns = Column(Integer, default=0, nullable=False)
    total_tokens = Column(Integer, default=0, nullable=False)
    total_cost_usd = Column(Numeric(10, 6), default=Decimal("0"), nullable=False)

    # Relationships
    turns = relationship("KrakenTurn", back_populates="conversation", cascade="all, delete-orphan")


class KrakenTurn(Base):
    """Model for kraken_turns table."""

    __tablename__ = "kraken_turns"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(PGUUID(as_uuid=True), ForeignKey("kraken_conversations.id"), nullable=False, index=True)
    turn_number = Column(Integer, nullable=False)
    user_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    cost_usd = Column(Numeric(10, 6), default=Decimal("0"), nullable=False)
    duration_ms = Column(Integer, default=0, nullable=False)
    tool_calls_count = Column(Integer, default=0, nullable=False)
    cache_creation_tokens = Column(Integer, default=0, nullable=False)
    cache_read_tokens = Column(Integer, default=0, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    conversation = relationship("KrakenConversation", back_populates="turns")
    tool_calls = relationship("KrakenToolCall", back_populates="turn", cascade="all, delete-orphan")


class KrakenToolCall(Base):
    """Model for kraken_tool_calls table."""

    __tablename__ = "kraken_tool_calls"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    turn_id = Column(PGUUID(as_uuid=True), ForeignKey("kraken_turns.id"), nullable=False, index=True)
    tool_name = Column(String, nullable=False)
    tool_args = Column(JSONB, nullable=False)
    tool_result = Column(JSONB, nullable=False)
    result_truncated = Column(Boolean, default=False, nullable=False)
    sequence_order = Column(Integer, nullable=False)

    # Relationships
    turn = relationship("KrakenTurn", back_populates="tool_calls")
