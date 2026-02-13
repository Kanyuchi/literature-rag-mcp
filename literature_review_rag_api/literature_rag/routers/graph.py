"""Knowledge graph router."""

import json
import logging
import os
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import load_config
from ..database import (
    get_db, JobCRUD, KnowledgeClaimCRUD,
    KnowledgeEntityCRUD, KnowledgeEdgeCRUD,
    KnowledgeEntityOccurrenceCRUD, KnowledgeClusterCRUD
)
from ..models import KnowledgeGraphResponse, KnowledgeGraphRunResponse, KnowledgeGraphClusterResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["KnowledgeGraph"])

config = load_config()


def _extract_json(content: str) -> list:
    try:
        return json.loads(content)
    except Exception:
        pass
    if "```" in content:
        cleaned = content.split("```", 1)[-1]
        cleaned = cleaned.split("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return []
    return []


def _extract_entities_relations(claim_text: str) -> dict:
    graph_cfg = getattr(config, "graph", None)
    provider = getattr(graph_cfg, "llm_provider", "openai")
    model = getattr(graph_cfg, "llm_model", "gpt-4.1-mini")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OPENAI_API_KEY not configured for graph extraction"
            )
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    else:
        groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Groq API key not configured for graph extraction"
            )
        from groq import Groq
        client = Groq(api_key=groq_api_key)

    prompt = (
        "Extract entities and relations from this claim. "
        "Return JSON with keys: entities (array of {name,type}) "
        "and relations (array of {source,target,relation}). "
        "Use short names, type in [concept, actor, institution, policy, place, theory].\n\n"
        f"Claim: {claim_text}\n\nJSON:"
    )

    response = client.chat.completions.create(
        model=model if provider == "openai" else config.llm.model,
        temperature=0.1,
        max_tokens=400,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    data = _extract_json(content)
    if not isinstance(data, dict):
        return {"entities": [], "relations": []}
    return {
        "entities": data.get("entities", []) or [],
        "relations": data.get("relations", []) or []
    }


def _refine_graph(raw_entities: List[Dict[str, Any]], raw_relations: List[Dict[str, Any]]) -> dict:
    graph_cfg = getattr(config, "graph", None)
    provider = getattr(graph_cfg, "llm_provider", "openai")
    model = getattr(graph_cfg, "llm_model", "gpt-4.1-mini")

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OPENAI_API_KEY not configured for graph refinement"
            )
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    else:
        groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Groq API key not configured for graph refinement"
            )
        from groq import Groq
        client = Groq(api_key=groq_api_key)

    trimmed_entities = raw_entities[:200]
    trimmed_relations = raw_relations[:300]

    prompt = (
        "You are refining a knowledge graph from extracted claims. "
        "Merge obvious duplicates and normalize names (short, consistent). "
        "Return JSON with keys: entities and relations. "
        "Each entity: {name, type, cluster}. "
        "Cluster is a short theme label (2-4 words) grouping related entities. "
        "Each relation: {source, target, relation}. "
        "Only keep relations where both entities exist. "
        "Avoid redundant edges.\n\n"
        f"ENTITIES: {json.dumps(trimmed_entities)}\n\n"
        f"RELATIONS: {json.dumps(trimmed_relations)}\n\n"
        "JSON:"
    )

    response = client.chat.completions.create(
        model=model if provider == "openai" else config.llm.model,
        temperature=0.1,
        max_tokens=800,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    data = _extract_json(content)
    if not isinstance(data, dict):
        return {"entities": raw_entities, "relations": raw_relations}
    refined_entities = data.get("entities", []) or []
    refined_relations = data.get("relations", []) or []
    if not refined_entities:
        refined_entities = raw_entities
    if not refined_relations:
        refined_relations = raw_relations
    return {"entities": refined_entities, "relations": refined_relations}


def _summarize_cluster(entities: List[str], relations: List[str]) -> str:
    graph_cfg = getattr(config, "graph", None)
    provider = getattr(graph_cfg, "llm_provider", "openai")
    model = getattr(graph_cfg, "llm_model", "gpt-4.1-mini")
    max_entities = getattr(graph_cfg, "cluster_summary_max_entities", 20)
    max_relations = getattr(graph_cfg, "cluster_summary_max_relations", 20)

    prompt = (
        "Summarize the following knowledge cluster in 1-2 sentences. "
        "Mention the dominant themes and key concepts.\n\n"
        f"Entities: {', '.join(entities[:max_entities])}\n"
        f"Relations: {', '.join(relations[:max_relations])}\n\nSummary:"
    )

    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not configured")
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "Return plain text only."},
                    {"role": "user", "content": prompt}
                ]
            )
        else:
            groq_api_key = config.llm.groq_api_key or os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise RuntimeError("Groq API key not configured")
            from groq import Groq
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model=config.llm.model,
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {"role": "system", "content": "Return plain text only."},
                    {"role": "user", "content": prompt}
                ]
            )
        return response.choices[0].message.content.strip()
    except Exception:
        if entities:
            return f"Cluster covering: {', '.join(entities[:6])}."
        return "Cluster of related concepts."


@router.post("/{job_id}/graph/build", response_model=KnowledgeGraphRunResponse)
async def build_knowledge_graph(
    job_id: int,
    claim_limit: int = Query(200, ge=1, le=1000),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = JobCRUD.get_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    claims = KnowledgeClaimCRUD.list_for_job(db, job_id, limit=claim_limit)
    KnowledgeEdgeCRUD.delete_for_job(db, job_id)
    KnowledgeEntityCRUD.delete_for_job(db, job_id)
    KnowledgeEntityOccurrenceCRUD.delete_for_job(db, job_id)
    KnowledgeClusterCRUD.delete_for_job(db, job_id)

    raw_entities: List[Dict[str, Any]] = []
    raw_relations: List[Dict[str, Any]] = []

    for claim in claims:
        extracted = _extract_entities_relations(claim.claim_text)
        for ent in extracted.get("entities", [])[:10]:
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(ent.get("type", "concept")).strip()
            raw_entities.append({
                "name": name,
                "type": entity_type,
                "doc_id": claim.doc_id,
                "claim_id": claim.id,
                "paragraph_index": claim.paragraph_index
            })

        for rel in extracted.get("relations", [])[:10]:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            relation_type = str(rel.get("relation", "related_to")).strip()
            if not source or not target:
                continue
            raw_relations.append({"source": source, "target": target, "relation": relation_type})

    refined = _refine_graph(raw_entities, raw_relations)
    entity_map: Dict[str, Any] = {}

    for ent in refined.get("entities", [])[:400]:
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        entity_type = str(ent.get("type", "concept")).strip()
        cluster = str(ent.get("cluster", "")).strip() or None
        entity = KnowledgeEntityCRUD.get_or_create(db, job_id, name, entity_type, cluster)
        entity_map[name] = entity

    for ent in raw_entities:
        name = str(ent.get("name", "")).strip()
        if not name or name not in entity_map:
            continue
        KnowledgeEntityOccurrenceCRUD.create(
            db=db,
            job_id=job_id,
            entity_id=entity_map[name].id,
            doc_id=str(ent.get("doc_id", "")),
            claim_id=ent.get("claim_id"),
            paragraph_index=ent.get("paragraph_index")
        )

    for rel in refined.get("relations", [])[:800]:
        source = str(rel.get("source", "")).strip()
        target = str(rel.get("target", "")).strip()
        relation_type = str(rel.get("relation", "related_to")).strip()
        if not source or not target:
            continue
        if source not in entity_map or target not in entity_map:
            continue
        KnowledgeEdgeCRUD.create(
            db=db,
            job_id=job_id,
            source_entity_id=entity_map[source].id,
            target_entity_id=entity_map[target].id,
            relation_type=relation_type,
            weight=1.0
        )

    graph_cfg = getattr(config, "graph", None)
    if getattr(graph_cfg, "cluster_summaries_enabled", True):
        # Build cluster summaries (connected components)
        entities = KnowledgeEntityCRUD.list_for_job(db, job_id, limit=1000)
        edges = KnowledgeEdgeCRUD.list_for_job(db, job_id, limit=2000)
        adjacency = {e.id: set() for e in entities}
        for edge in edges:
            adjacency.setdefault(edge.source_entity_id, set()).add(edge.target_entity_id)
            adjacency.setdefault(edge.target_entity_id, set()).add(edge.source_entity_id)

        visited = set()
        clusters = []
        for entity in entities:
            if entity.id in visited:
                continue
            stack = [entity.id]
            component = []
            while stack:
                node_id = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                component.append(node_id)
                for neighbor in adjacency.get(node_id, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
            if component:
                clusters.append(component)

        for idx, component in enumerate(clusters, start=1):
            cluster_id = f"cluster_{idx}"
            cluster_entities = [e for e in entities if e.id in component]
            entity_names = [e.name for e in cluster_entities]
            relation_names = [
                edge.relation_type
                for edge in edges
                if edge.source_entity_id in component or edge.target_entity_id in component
            ]
            summary = _summarize_cluster(entity_names, relation_names)
            KnowledgeClusterCRUD.upsert(
                db=db,
                job_id=job_id,
                cluster_id=cluster_id,
                name=entity_names[0] if entity_names else cluster_id,
                summary=summary,
                node_count=len(component)
            )
            for e in cluster_entities:
                if not e.cluster:
                    e.cluster = cluster_id
            db.commit()

    return {
        "claims_processed": len(claims),
        "entities_created": KnowledgeEntityCRUD.count_for_job(db, job_id),
        "edges_created": KnowledgeEdgeCRUD.count_for_job(db, job_id)
    }


@router.get("/{job_id}/graph", response_model=KnowledgeGraphResponse)
async def get_knowledge_graph(
    job_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = JobCRUD.get_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    entities = KnowledgeEntityCRUD.list_for_job(db, job_id, limit=1000)
    edges = KnowledgeEdgeCRUD.list_for_job(db, job_id, limit=2000)

    return {
        "nodes": [
            {"id": e.id, "name": e.name, "entity_type": e.entity_type, "cluster": e.cluster}
            for e in entities
        ],
        "edges": [
            {"source": e.source_entity_id, "target": e.target_entity_id, "relation_type": e.relation_type, "weight": e.weight}
            for e in edges
        ],
        "clusters": [
            {"cluster_id": c.cluster_id, "name": c.name, "summary": c.summary, "node_count": c.node_count}
            for c in clusters
        ]
    }


@router.get("/{job_id}/graph/clusters", response_model=KnowledgeGraphClusterResponse)
async def get_graph_clusters(
    job_id: int,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db)
):
    job = JobCRUD.get_by_id(db, job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    clusters = KnowledgeClusterCRUD.list_for_job(db, job_id)
    return {
        "clusters": [
            {"cluster_id": c.cluster_id, "name": c.name, "summary": c.summary, "node_count": c.node_count}
            for c in clusters
        ]
    }
