from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, UploadFile
from fastapi import Path as PathParam

from app.core.config import settings
from app.knowledge_base.embedder import ChunkEmbedder
from app.knowledge_base.service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])

_UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"


def _get_service(request: Request) -> KnowledgeBaseService:
    # TODO(M4): move to lifespan — instantiate once and cache on app.state
    state = request.app.state
    try:
        return KnowledgeBaseService(
            pg=state.postgres,
            redis=state.redis,
            milvus=state.milvus,
            mq=state.mq,
            embedder=ChunkEmbedder(llm=state.llm, batch_size=settings.embedding_batch_size),
        )
    except AttributeError as exc:
        raise HTTPException(status_code=503, detail="Service dependencies not ready") from exc


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    # TODO(M4): add authentication dependency
    svc = _get_service(request)
    doc_id, tmp_path = await svc.prepare_upload(file)
    background_tasks.add_task(svc.run_ingest, doc_id, tmp_path)
    return {"doc_id": doc_id}


@router.get("/documents/{doc_id}/status")
async def get_document_status(
    request: Request,
    doc_id: str = PathParam(..., pattern=_UUID_PATTERN),
) -> dict:
    # TODO(M4): add authentication dependency
    return await _get_service(request).get_document_status(doc_id)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(
    request: Request,
    doc_id: str = PathParam(..., pattern=_UUID_PATTERN),
) -> None:
    # TODO(M4): add authentication dependency
    await _get_service(request).delete_document(doc_id)
