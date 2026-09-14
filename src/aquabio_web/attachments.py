"""Upload validation and attachment context extraction."""
from __future__ import annotations
import hashlib
import io
import mimetypes
import uuid
from pathlib import Path
from typing import Any
import fitz
from PIL import Image
from aquabio_mrag.conversation import ConversationStore

UPLOAD_LIMITS = {"image": 20 * 1024 * 1024, "pdf": 50 * 1024 * 1024}


class AttachmentService:
    def save_upload(
        self,
        session_id: str,
        file_name: str,
        content: bytes,
        expected_type: str,
    ) -> dict[str, Any]:
        ConversationStore.normalize_session_id(session_id)
        file_name = Path(file_name.replace("\\", "/")).name
        if expected_type not in UPLOAD_LIMITS:
            raise ValueError("Unsupported upload type")
        if not content or len(content) > UPLOAD_LIMITS[expected_type]:
            raise ValueError("文件为空或超过大小限制")
        suffix = Path(file_name).suffix.lower()
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        pdf_suffixes = {".pdf"}
        allowed = image_suffixes if expected_type == "image" else pdf_suffixes
        if suffix not in allowed:
            raise ValueError(f"不支持的文件类型：{suffix}")
        file_id = f"file_{uuid.uuid4().hex}"
        target_dir = self.paths.uploads_dir / expected_type
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{file_id}{suffix}"
        metadata: dict[str, Any] = {"mime_type": mimetypes.guess_type(file_name)[0] or ""}
        try:
            if expected_type == "pdf":
                with fitz.open(stream=content, filetype="pdf") as document:
                    if document.needs_pass:
                        raise ValueError("请上传未加密的 PDF")
                    metadata["page_count"] = document.page_count
            else:
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
        except Exception as error:
            raise ValueError("文件内容不是有效的图片或 PDF") from error
        target.write_bytes(content)
        relative = str(target.relative_to(self.root)).replace("\\", "/")
        try:
            row = self.store.save_attachment(
                session_id,
                file_id,
                expected_type,
                file_name,
                relative,
                len(content),
                metadata,
            )
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        row["url"] = f"/files/{relative}"
        return row

    def _pdf_context(self, attachments: list[dict[str, Any]]) -> str:
        sections = []
        for attachment in attachments:
            if attachment["file_type"] != "pdf":
                continue
            path = self.root / attachment["file_path"]
            document = fitz.open(path)
            try:
                pages = []
                total = 0
                for page_index, page in enumerate(document):
                    text = " ".join(page.get_text("text").split())
                    if not text:
                        continue
                    excerpt = text[:4000]
                    pages.append(
                        f"[UPLOADED_PDF={attachment['file_name']}]"
                        f"[PAGE={page_index + 1}]\n{excerpt}"
                    )
                    total += len(excerpt)
                    if total >= 16000:
                        break
                sections.extend(pages)
            finally:
                document.close()
        return "\n\n".join(sections)

    def _image_path(
        self, attachment: dict[str, Any]
    ) -> str:
        uploaded = (self.root / attachment["file_path"]).resolve()
        original_name = attachment.get("file_name", "")
        if not original_name or not uploaded.is_file():
            return str(uploaded)
        uploaded_hash = hashlib.sha256(
            uploaded.read_bytes()
        ).digest()
        for candidate in self.paths.images_dir.rglob(original_name):
            if (
                candidate.is_file()
                and candidate.stat().st_size == uploaded.stat().st_size
                and hashlib.sha256(candidate.read_bytes()).digest()
                == uploaded_hash
            ):
                return str(candidate.resolve())
        return str(uploaded)

