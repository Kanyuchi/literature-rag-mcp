"""Chat session persistence and export endpoints."""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db, JobCRUD, ChatSessionCRUD, ChatMessageCRUD
from ..models import (
    ChatSessionCreateRequest,
    ChatSessionResponse,
    ChatSessionListResponse,
    ChatMessageCreateRequest,
    ChatMessageResponse,
    ChatSessionDetailResponse,
)

router = APIRouter(prefix="/api/chats", tags=["chat_sessions"])


@router.post("", response_model=ChatSessionResponse)
async def create_chat_session(
    request: ChatSessionCreateRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = JobCRUD.get_by_id(db, request.job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    session = ChatSessionCRUD.create(db, current_user.id, request.job_id, request.title)
    return session


@router.get("", response_model=ChatSessionListResponse)
async def list_chat_sessions(
    job_id: int = Query(..., description="Knowledge base (job) id"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = ChatSessionCRUD.list_for_job(db, current_user.id, job_id)
    return ChatSessionListResponse(total=len(sessions), sessions=sessions)


@router.get("/{session_id}", response_model=ChatSessionDetailResponse)
async def get_chat_session(
    session_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = ChatSessionCRUD.get_by_id(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    messages = ChatMessageCRUD.list_for_session(db, session_id)
    response_messages = []
    for msg in messages:
        citations = None
        if msg.citations_json:
            try:
                citations = json.loads(msg.citations_json)
            except Exception:
                citations = None
        response_messages.append(
            ChatMessageResponse(
                id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                content=msg.content,
                citations=citations,
                model=msg.model,
                created_at=msg.created_at.isoformat(),
            )
        )

    return ChatSessionDetailResponse(session=session, messages=response_messages)


@router.post("/{session_id}/messages", response_model=ChatMessageResponse)
async def add_chat_message(
    session_id: int,
    request: ChatMessageCreateRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = ChatSessionCRUD.get_by_id(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    citations_json = json.dumps(request.citations) if request.citations is not None else None
    message = ChatMessageCRUD.create(
        db,
        session_id=session_id,
        role=request.role,
        content=request.content,
        citations_json=citations_json,
        model=request.model,
    )

    ChatSessionCRUD.touch(db, session)

    return ChatMessageResponse(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        citations=request.citations,
        model=message.model,
        created_at=message.created_at.isoformat(),
    )


@router.get("/{session_id}/export")
async def export_chat_session(
    session_id: int,
    format: str = Query("md", description="Export format: md"),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if format.lower() != "md":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported format")

    session = ChatSessionCRUD.get_by_id(db, session_id)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found")

    messages = ChatMessageCRUD.list_for_session(db, session_id)

    lines = []
    title = session.title or f"Chat Session {session.id}"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Job ID: {session.job_id}")
    lines.append(f"- Created: {session.created_at.isoformat()}")
    lines.append(f"- Updated: {session.updated_at.isoformat()}")
    lines.append("")

    for msg in messages:
        role = msg.role.capitalize()
        timestamp = msg.created_at.isoformat()
        lines.append(f"## {role} ({timestamp})")
        lines.append("")
        lines.append(msg.content)
        lines.append("")
        if msg.citations_json:
            try:
                citations = json.loads(msg.citations_json)
            except Exception:
                citations = None
            if citations:
                lines.append("**References:**")
                for idx, citation in enumerate(citations, start=1):
                    title = citation.get("title") or "Untitled"
                    authors = citation.get("authors") or "Unknown"
                    year = citation.get("year") or "n.d."
                    lines.append(f"[{idx}] {authors} ({year}). {title}")
                lines.append("")

    markdown = "\n".join(lines).strip() + "\n"
    filename = f"chat_session_{session.id}.md"
    return Response(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
