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
    user_id varchar(64) not null,
    domain varchar(32) not null,
    tool_name varchar(128) not null,
    parameter_summary text null,
    status varchar(32) not null,
    duration_ms bigint not null,
    created_at datetime(6) not null,
    key idx_agent_tool_audit_user_created (user_id, created_at),
    key idx_agent_tool_audit_request (request_id)
);

create table if not exists unknown_intent_audit (
    id bigint not null auto_increment primary key,
    request_id varchar(64) not null,
    user_id varchar(64) not null,
    domain varchar(32) not null,
    user_message text not null,
    raw_intent_json text null,
    classifier_intent varchar(64) not null,
    embedding_top1_intent varchar(64) null,
    embedding_top2_intent varchar(64) null,
    reason text not null,
    semantic_score double not null,
    margin_score double not null,
    agreement_score double not null,
    final_confidence double not null,
    fallback_cause varchar(64) not null,
    created_at datetime(6) not null,
    key idx_unknown_intent_audit_created (created_at),
    key idx_unknown_intent_audit_request (request_id)
);

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
);

create table if not exists knowledge_document_chunk (
    document_id varchar(191) not null,
    chunk_id varchar(64) not null,
    primary key (document_id, chunk_id),
    constraint fk_knowledge_document_chunk_document
        foreign key (document_id) references knowledge_document(document_id) on delete cascade
);

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
);

create table if not exists agent_eval_run (
    id bigint not null auto_increment primary key,
    schema_version int not null default 2,
    suite_name varchar(128) not null,
    total_cases int not null,
    passed_cases int not null,
    score double not null,
    details_json longtext null,
    created_at datetime(6) not null,
    key idx_agent_eval_run_suite_created (suite_name, created_at)
);

create table if not exists agent_trace_feedback (
    id bigint not null auto_increment primary key,
    trace_id varchar(64) not null,
    request_id varchar(64) not null,
    rating int not null,
    comment text null,
    created_at datetime(6) not null,
    key idx_agent_trace_feedback_trace (trace_id),
    key idx_agent_trace_feedback_request (request_id)
);
