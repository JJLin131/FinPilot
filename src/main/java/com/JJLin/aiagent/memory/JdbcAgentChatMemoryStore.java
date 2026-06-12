package com.JJLin.aiagent.memory;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageDeserializer;
import dev.langchain4j.data.message.ChatMessageSerializer;
import dev.langchain4j.store.memory.chat.ChatMemoryStore;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class JdbcAgentChatMemoryStore implements ChatMemoryStore {

    private final JdbcTemplate jdbcTemplate;

    public JdbcAgentChatMemoryStore(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    @Override
    public List<ChatMessage> getMessages(Object memoryId) {
        List<String> rows = jdbcTemplate.query(
                "select messages_json from agent_chat_memory where memory_id = ?",
                (rs, rowNum) -> rs.getString(1),
                String.valueOf(memoryId));
        return rows.isEmpty() ? List.of() : ChatMessageDeserializer.messagesFromJson(rows.get(0));
    }

    @Override
    public void updateMessages(Object memoryId, List<ChatMessage> messages) {
        jdbcTemplate.update("""
                insert into agent_chat_memory(memory_id, messages_json, updated_at)
                values (?, ?, current_timestamp(6))
                on duplicate key update messages_json = values(messages_json), updated_at = current_timestamp(6)
                """, String.valueOf(memoryId), ChatMessageSerializer.messagesToJson(messages));
    }

    @Override
    public void deleteMessages(Object memoryId) {
        jdbcTemplate.update("delete from agent_chat_memory where memory_id = ?", String.valueOf(memoryId));
    }
}
