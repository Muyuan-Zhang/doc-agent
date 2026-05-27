from fastapi import APIRouter, BackgroundTasks, Request, UploadFile

from app.core.config import settings
from app.knowledge_base.embedder import ChunkEmbedder
from app.knowledge_base.service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-base", tags=["knowledge-base"])


def _get_service(request: Request) -> KnowledgeBaseService:
    state = request.app.state
    return KnowledgeBaseService(
        pg=state.postgres,
        redis=state.redis,
        milvus=state.milvus,
        mq=state.mq,
        embedder=ChunkEmbedder(llm=state.llm, batch_size=settings.embedding_batch_size),
    )


@router.post("/documents", status_code=202)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    request: Request,
) -> dict:
    svc = _get_service(request)
    doc_id, tmp_path = await svc.prepare_upload(file)
    background_tasks.add_task(svc.run_ingest, doc_id, tmp_path)
    return {"doc_id": doc_id}


@router.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str, request: Request) -> dict:
    return await _get_service(request).get_document_status(doc_id)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, request: Request) -> None:
    await _get_service(request).delete_document(doc_id)
