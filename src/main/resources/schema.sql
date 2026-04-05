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
