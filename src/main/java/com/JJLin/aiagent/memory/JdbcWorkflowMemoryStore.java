package com.JJLin.aiagent.memory;

import com.JJLin.aiagent.entites.WorkflowMemory;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.time.Instant;
import java.util.List;
import java.util.Optional;

@Repository
public class JdbcWorkflowMemoryStore implements WorkflowMemoryStore {

    private static final TypeReference<List<String>> STRING_LIST_TYPE = new TypeReference<>() {
    };

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper;

    public JdbcWorkflowMemoryStore(JdbcTemplate jdbcTemplate, ObjectMapper objectMapper) {
        this.jdbcTemplate = jdbcTemplate;
        this.objectMapper = objectMapper;
    }

    @Override
    public Optional<WorkflowMemory> find(String sessionId) {
        List<WorkflowMemory> rows = jdbcTemplate.query(
                """
                        select session_id,
                               resolved_facts,
                               open_questions,
                               latest_plan_summary,
                               latest_execution_summary,
                               last_failure_tag,
                               last_required_fixes,
                               updated_at
                        from workflow_memory
                        where session_id = ?
                        """,
                (rs, rowNum) -> mapRow(rs),
                sessionId);
        return rows.stream().findFirst();
    }

    @Override
    public WorkflowMemory save(WorkflowMemory workflowMemory) {
        jdbcTemplate.update(
                """
                        insert into workflow_memory (
                            session_id,
                            resolved_facts,
                            open_questions,
                            latest_plan_summary,
                            latest_execution_summary,
                            last_failure_tag,
                            last_required_fixes,
                            updated_at
                        ) values (?, ?, ?, ?, ?, ?, ?, ?)
                        on duplicate key update
                            resolved_facts = values(resolved_facts),
                            open_questions = values(open_questions),
                            latest_plan_summary = values(latest_plan_summary),
                            latest_execution_summary = values(latest_execution_summary),
                            last_failure_tag = values(last_failure_tag),
                            last_required_fixes = values(last_required_fixes),
                            updated_at = values(updated_at)
                        """,
                workflowMemory.getSessionId(),
                toJson(workflowMemory.getResolvedFacts()),
                toJson(workflowMemory.getOpenQuestions()),
                workflowMemory.getLatestPlanSummary(),
                workflowMemory.getLatestExecutionSummary(),
                workflowMemory.getLastFailureTag() == null ? null : workflowMemory.getLastFailureTag().name(),
                toJson(workflowMemory.getLastRequiredFixes()),
                Timestamp.from(defaultUpdatedAt(workflowMemory)));
        return workflowMemory;
    }

    @Override
    public void clear(String sessionId) {
        jdbcTemplate.update("delete from workflow_memory where session_id = ?", sessionId);
    }

    private WorkflowMemory mapRow(ResultSet rs) throws SQLException {
        Timestamp updatedAt = rs.getTimestamp("updated_at");
        String lastFailureTag = rs.getString("last_failure_tag");
        return WorkflowMemory.builder()
                .sessionId(rs.getString("session_id"))
                .resolvedFacts(readJsonList(rs.getString("resolved_facts")))
                .openQuestions(readJsonList(rs.getString("open_questions")))
                .latestPlanSummary(rs.getString("latest_plan_summary"))
                .latestExecutionSummary(rs.getString("latest_execution_summary"))
                .lastFailureTag(lastFailureTag == null || lastFailureTag.isBlank() ? null : VerificationFailureTag.valueOf(lastFailureTag))
                .lastRequiredFixes(readJsonList(rs.getString("last_required_fixes")))
                .updatedAt(updatedAt == null ? Instant.now() : updatedAt.toInstant())
                .build();
    }

    private Instant defaultUpdatedAt(WorkflowMemory workflowMemory) {
        return workflowMemory.getUpdatedAt() == null ? Instant.now() : workflowMemory.getUpdatedAt();
    }

    private String toJson(List<String> values) {
        try {
            return objectMapper.writeValueAsString(values == null ? List.of() : values);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to serialize workflow memory field.", exception);
        }
    }

    private List<String> readJsonList(String rawValue) {
        if (rawValue == null || rawValue.isBlank()) {
            return List.of();
        }
        try {
            return objectMapper.readValue(rawValue, STRING_LIST_TYPE);
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("Failed to deserialize workflow memory field.", exception);
        }
    }
}