from fastapi import APIRouter, Depends, HTTPException

from app.schemas import DocumentInfo
from app.services import vectorstore
from app.services.auth_service import User, get_current_user

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=dict)
async def list_documents(user: User = Depends(get_current_user)):
    docs = [DocumentInfo(**d) for d in vectorstore.list_documents(user.id)]
    return {"documents": docs}


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, user: User = Depends(get_current_user)):
    deleted = vectorstore.delete_document(user.id, doc_id)
    if not deleted:
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}
