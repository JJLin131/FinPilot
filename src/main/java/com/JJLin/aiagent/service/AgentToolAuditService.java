package com.JJLin.aiagent.service;

import com.JJLin.aiagent.agent.AgentRequestContext;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class AgentToolAuditService {

    private final JdbcTemplate jdbcTemplate;

    public AgentToolAuditService(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public void record(String toolName, String parameterSummary, String status, long durationMs) {
        AgentRequestContext.State context = AgentRequestContext.require();
        jdbcTemplate.update("""
                insert into agent_tool_audit
                (request_id, tenant_id, user_id, domain, tool_name, parameter_summary, status, duration_ms, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, current_timestamp(6))
                """, context.requestId(), context.tenantId(), context.userId(), context.domain(), toolName,
                parameterSummary, status, durationMs);
    }
}
