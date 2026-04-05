package com.JJLin.aiagent.extractor;

import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.entites.UserProfilePatch;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;

@Service
public class UserProfileExtractorService {

    private final ChatClient profileExtractorAgent;

    public UserProfileExtractorService(@Qualifier("profileExtractorAgent") ChatClient profileExtractorAgent) {
        this.profileExtractorAgent = profileExtractorAgent;
    }

    public UserProfilePatch extract(String content, UserProfile existingProfile) {
        if (content == null || content.isBlank()) {
            return new UserProfilePatch();
        }
        UserProfilePatch patch = profileExtractorAgent.prompt()
                .user(buildPrompt(content, existingProfile))
                .call()
                .entity(UserProfilePatch.class);
        return patch == null ? new UserProfilePatch() : patch;
    }

    private String buildPrompt(String content, UserProfile existingProfile) {
        StringBuilder prompt = new StringBuilder();
        prompt.append("Current user message: ").append(content).append("\n");
        if (existingProfile != null) {
            prompt.append("Existing profile: ")
                    .append(existingProfile)
                    .append("\n");
        }
        prompt.append("Extract only explicit profile fields from the current user message. ")
                .append("Allowed fields: age, occupation, education, incomeRange, gender, city, maritalStatus, notes. ")
                .append("Do not infer unspecified fields. ")
                .append("If a field is not explicitly stated, omit it from the JSON output.");
        return prompt.toString();
    }
}
