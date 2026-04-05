package com.JJLin.aiagent.extractor;

import com.JJLin.aiagent.entites.UserLongTermMemoryExtractionResult;
import com.JJLin.aiagent.entites.UserProfile;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class UserLongTermMemoryExtractorService {

    private final ChatClient longTermMemoryExtractorAgent;

    public UserLongTermMemoryExtractorService(@Qualifier("longTermMemoryExtractorAgent") ChatClient longTermMemoryExtractorAgent) {
        this.longTermMemoryExtractorAgent = longTermMemoryExtractorAgent;
    }

    public List<String> extract(String userId, String content, UserProfile existingProfile) {
        if (userId == null || userId.isBlank() || content == null || content.isBlank()) {
            return List.of();
        }
        UserLongTermMemoryExtractionResult result = longTermMemoryExtractorAgent.prompt()
                .user(buildPrompt(content, existingProfile))
                .call()
                .entity(UserLongTermMemoryExtractionResult.class);
        if (result == null || result.isEmpty()) {
            return List.of();
        }
        return result.getMemories().stream()
                .filter(value -> value != null && !value.isBlank())
                .map(String::trim)
                .distinct()
                .toList();
    }

    private String buildPrompt(String content, UserProfile existingProfile) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Current user message: ").append(content).append("\n");
        if (existingProfile != null) {
            prompt.append("Existing structured profile: ")
                    .append(existingProfile)
                    .append("\n");
        }
        prompt.append("Extract durable long-term user memories from this single message. ")
                .append("You may summarize explicit preferences, budgets, constraints, shopping scenarios, gift recipients, and user background. ")
                .append("Do not include system/tool information. ")
                .append("Do not repeat information already stored as structured profile fields. ")
                .append("Return a JSON object with one field: memories, which is a list of concise strings. ")
                .append("If nothing should be stored, return {\"memories\": []}.");
        return prompt.toString();
    }
}
