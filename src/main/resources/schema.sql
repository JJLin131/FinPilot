create table if not exists workflow_memory (
    session_id varchar(191) not null primary key,
    resolved_facts longtext null,
    open_questions longtext null,
    latest_plan_summary longtext null,
    latest_execution_summary longtext null,
    last_failure_tag varchar(64) null,
    last_required_fixes longtext null,
    updated_at datetime(6) not null
);

create table if not exists user_profile (
    user_id varchar(64) not null primary key,
    age int null,
    occupation varchar(128) null,
    education varchar(64) null,
    income_range varchar(64) null,
    gender varchar(32) null,
    city varchar(64) null,
    marital_status varchar(32) null,
    notes varchar(255) null,
    created_at datetime(6) not null,
    updated_at datetime(6) not null
);

create table if not exists system_operation_memory (
    id bigint not null auto_increment primary key,
    user_id varchar(64) not null,
    operation_type varchar(64) not null,
    content text not null,
    created_at datetime(6) not null,
    key idx_system_operation_memory_user_operation_created (user_id, operation_type, created_at)
);

create table if not exists agent_chat_memory (
    memory_id varchar(255) not null primary key,
    messages_json longtext not null,
    updated_at datetime(6) not null
);

create table if not exists agent_tool_audit (
    id bigint not null auto_increment primary key,
    request_id varchar(64) not null,
    tenant_id varchar(128) null,
    user_id varchar(64) not null,
    domain varchar(32) not null,
    tool_name varchar(128) not null,
    parameter_summary text null,
    status varchar(32) not null,
    duration_ms bigint not null,
    created_at datetime(6) not null,
    key idx_agent_tool_audit_tenant_created (tenant_id, created_at),
    key idx_agent_tool_audit_request (request_id)
);

create table if not exists knowledge_document (
    document_id varchar(191) not null primary key,
    domain varchar(32) not null,
    tenant_id varchar(128) null,
    title varchar(255) not null,
    source_uri varchar(500) not null,
    content_hash varchar(64) not null,
    tags_text text null,
    status varchar(32) not null,
    valid_from date null,
    valid_to date null,
    chunk_count int not null,
    updated_at datetime(6) not null,
    key idx_knowledge_document_scope_status (domain, tenant_id, status),
    key idx_knowledge_document_expiry (status, valid_to)
);

create table if not exists knowledge_document_chunk (
    document_id varchar(191) not null,
    chunk_id varchar(64) not null,
    primary key (document_id, chunk_id),
    constraint fk_knowledge_document_chunk_document
        foreign key (document_id) references knowledge_document(document_id) on delete cascade
);
