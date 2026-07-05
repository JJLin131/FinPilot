from __future__ import annotations

import hashlib
from datetime import date

import pymysql

from finpilot.mysql import connect_runtime_mysql
from finpilot.rag.models import KnowledgeDocumentRecord, KnowledgeDocumentRequest


class KnowledgeDocumentRegistry:
    def __init__(self):
        self._ensure_schema()

    def _connect(self):
        return connect_runtime_mysql()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists knowledge_document (
                        document_id varchar(191) not null primary key,
                        domain varchar(32) not null,
                        title varchar(255) not null,
                        source_uri varchar(500) not null,
                        content_hash varchar(64) not null,
                        tags_text text null,
                        status varchar(32) not null,
                        valid_from date null,
                        valid_to date null,
                        chunk_count int not null,
                        updated_at datetime(6) not null,
                        key idx_knowledge_document_scope_status (domain, status),
                        key idx_knowledge_document_expiry (status, valid_to)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists knowledge_document_chunk (
                        document_id varchar(191) not null,
                        chunk_id varchar(64) not null,
                        primary key (document_id, chunk_id),
                        constraint fk_knowledge_document_chunk_document
                            foreign key (document_id) references knowledge_document(document_id) on delete cascade
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists knowledge_chunk_content (
                        chunk_id varchar(64) not null primary key,
                        document_id varchar(191) not null,
                        domain varchar(32) not null,
                        title varchar(255) not null,
                        source_uri varchar(500) not null,
                        tags_text text null,
                        chunk_index int not null,
                        content longtext not null,
                        created_at datetime(6) not null,
                        updated_at datetime(6) not null,
                        key idx_knowledge_chunk_doc (document_id),
                        key idx_knowledge_chunk_scope (domain)
                    )
                    """
                )

    def save_document(self, request: KnowledgeDocumentRequest, chunk_count: int) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into knowledge_document
                    (document_id, domain, title, source_uri, content_hash, tags_text,
                     status, valid_from, valid_to, chunk_count, updated_at)
                    values (%s, %s, %s, %s, %s, %s, 'ACTIVE', %s, %s, %s, current_timestamp(6))
                    on duplicate key update
                        domain = values(domain),
                        title = values(title),
                        source_uri = values(source_uri),
                        content_hash = values(content_hash),
                        tags_text = values(tags_text),
                        status = 'ACTIVE',
                        valid_from = values(valid_from),
                        valid_to = values(valid_to),
                        chunk_count = values(chunk_count),
                        updated_at = current_timestamp(6)
                    """,
                    (
                        request.document_id,
                        request.domain,
                        request.title,
                        request.source,
                        hashlib.sha256(request.content.encode("utf-8")).hexdigest(),
                        ",".join(request.tags),
                        request.valid_from,
                        request.valid_to,
                        chunk_count,
                    ),
                )

    def replace_chunks(
        self,
        request: KnowledgeDocumentRequest,
        chunks: list[str],
        chunk_ids: list[str] | None = None,
    ) -> list[str]:
        resolved_chunk_ids = chunk_ids or self.chunk_ids_for(request.document_id, len(chunks))
        if len(resolved_chunk_ids) != len(chunks):
            raise ValueError("chunk_ids length must match chunks length.")
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("delete from knowledge_document_chunk where document_id = %s", (request.document_id,))
                cursor.execute("delete from knowledge_chunk_content where document_id = %s", (request.document_id,))
                for index, text in enumerate(chunks):
                    chunk_id = resolved_chunk_ids[index]
                    cursor.execute(
                        "insert into knowledge_document_chunk(document_id, chunk_id) values (%s, %s)",
                        (request.document_id, chunk_id),
                    )
                    cursor.execute(
                        """
                        insert into knowledge_chunk_content
                        (chunk_id, document_id, domain, title, source_uri, tags_text, chunk_index, content, created_at, updated_at)
                        values (%s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6), current_timestamp(6))
                        """,
                        (
                            chunk_id,
                            request.document_id,
                            request.domain,
                            request.title,
                            request.source,
                            ",".join(request.tags),
                            index,
                            text,
                        ),
                    )
        return resolved_chunk_ids

    def chunk_ids(self, document_id: str) -> list[str]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select chunk_id from knowledge_document_chunk where document_id = %s", (document_id,))
                rows = cursor.fetchall()
        return [row[0] for row in rows]

    def is_active(self, document_id: str, domain: str, today: date) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select 1 from knowledge_document
                    where document_id = %s
                      and domain = %s
                      and status = 'ACTIVE'
                      and (valid_from is null or valid_from <= %s)
                      and (valid_to is null or valid_to >= %s)
                    limit 1
                    """,
                    (document_id, domain, today, today),
                )
                return cursor.fetchone() is not None

    def domain(self, document_id: str) -> str:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select domain from knowledge_document where document_id = %s", (document_id,))
                row = cursor.fetchone()
        return row[0] if row else "FINANCE"

    def documents_by_ids(self, document_ids: list[str]) -> list[KnowledgeDocumentRecord]:
        if not document_ids:
            return []
        placeholders = ",".join(["%s"] * len(document_ids))
        with self._connect() as connection:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(
                    f"""
                    select document_id, domain, title, status, valid_from, valid_to
                    from knowledge_document
                    where document_id in ({placeholders})
                    """,
                    tuple(document_ids),
                )
                rows = cursor.fetchall()
        return [KnowledgeDocumentRecord.model_validate(row) for row in rows]

    def chunk_texts(self, domain: str, today: date) -> list[dict]:
        with self._connect() as connection:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(
                    """
                    select kc.chunk_id, kc.document_id, kc.title, kc.source_uri, kc.tags_text, kc.content
                    from knowledge_chunk_content kc
                    join knowledge_document kd on kd.document_id = kc.document_id
                    where kd.domain = %s and kd.status = 'ACTIVE'
                      and (kd.valid_from is null or kd.valid_from <= %s)
                      and (kd.valid_to is null or kd.valid_to >= %s)
                    """,
                    (domain, today, today),
                )
                return list(cursor.fetchall())

    def expired_document_ids(self, today: date) -> list[str]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select document_id from knowledge_document
                    where status = 'ACTIVE' and valid_to is not null and valid_to < %s
                    """,
                    (today,),
                )
                rows = cursor.fetchall()
        return [row[0] for row in rows]

    def mark_expired(self, document_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "update knowledge_document set status='EXPIRED', updated_at=current_timestamp(6) where document_id = %s",
                    (document_id,),
                )

    def delete_document_chunks(self, document_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("delete from knowledge_document_chunk where document_id = %s", (document_id,))
                cursor.execute("delete from knowledge_chunk_content where document_id = %s", (document_id,))

    def chunk_ids_for(self, document_id: str, chunk_count: int) -> list[str]:
        return [self._chunk_id(document_id, index) for index in range(chunk_count)]

    def _chunk_id(self, document_id: str, index: int) -> str:
        return hashlib.sha1(f"{document_id}:{index}".encode("utf-8")).hexdigest()[:32]

