package com.JJLin.aiagent.extractor;

import com.JJLin.aiagent.entites.UserLongTermMemoryExtractionResult;
import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.entites.UserProfilePatch;
import org.junit.jupiter.api.Test;
import org.mockito.Answers;
import org.springframework.ai.chat.client.ChatClient;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class UserMemoryExtractorServiceTest {

    @Test
    void shouldExtractExplicitUserProfileFieldsFromCurrentTurn() {
        ChatClient profileExtractorAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        UserProfilePatch patch = new UserProfilePatch();
        patch.setAge(28);
        patch.setOccupation("\u8001\u5e08");
        patch.setCity("\u4e0a\u6d77");
        patch.setEducation("\u672c\u79d1");
        patch.setIncomeRange("10k");
        when(profileExtractorAgent.prompt().user(anyString()).call().entity(UserProfilePatch.class)).thenReturn(patch);

        UserProfileExtractorService userProfileExtractorService = new UserProfileExtractorService(profileExtractorAgent);
        UserProfilePatch result = userProfileExtractorService.extract("\u621128\u5c81\uff0c\u6211\u662f\u8001\u5e08\uff0c\u6765\u81ea\u4e0a\u6d77\uff0c\u672c\u79d1\uff0c\u6708\u516510k", null);

        assertEquals(28, result.getAge());
        assertEquals("\u8001\u5e08", result.getOccupation());
        assertEquals("\u4e0a\u6d77", result.getCity());
        assertEquals("\u672c\u79d1", result.getEducation());
        assertEquals("10k", result.getIncomeRange());
    }

    @Test
    void shouldExtractIncrementalLongTermUserMemoryFromCurrentTurnOnly() {
        ChatClient longTermMemoryExtractorAgent = mock(ChatClient.class, Answers.RETURNS_DEEP_STUBS);
        UserLongTermMemoryExtractionResult extractionResult = new UserLongTermMemoryExtractionResult();
        extractionResult.setMemories(List.of(
                "\u7528\u6237\u504f\u597d\uff1a\u8010\u514b",
                "\u7528\u6237\u9884\u7b97\uff1a500\u5143",
                "\u7528\u6237\u7ea6\u675f\uff1a\u8dd1\u978b",
                "\u7528\u6237\u573a\u666f\uff1a\u7ed9\u7238\u7238\u4e70\u793c\u7269"));
        when(longTermMemoryExtractorAgent.prompt().user(anyString()).call().entity(UserLongTermMemoryExtractionResult.class))
                .thenReturn(extractionResult);

        UserLongTermMemoryExtractorService userLongTermMemoryExtractorService =
                new UserLongTermMemoryExtractorService(longTermMemoryExtractorAgent);
        UserProfile profile = new UserProfile();
        profile.setOccupation("\u8001\u5e08");

        List<String> memories = userLongTermMemoryExtractorService.extract(
                "u-1",
                "\u6211\u559c\u6b22\u8010\u514b\uff0c\u9884\u7b97500\u5143\uff0c\u53ea\u770b\u8dd1\u978b\uff0c\u7ed9\u7238\u7238\u4e70\u793c\u7269",
                profile);

        assertTrue(memories.contains("\u7528\u6237\u504f\u597d\uff1a\u8010\u514b"));
        assertTrue(memories.contains("\u7528\u6237\u9884\u7b97\uff1a500\u5143"));
        assertTrue(memories.contains("\u7528\u6237\u7ea6\u675f\uff1a\u8dd1\u978b"));
        assertTrue(memories.contains("\u7528\u6237\u573a\u666f\uff1a\u7ed9\u7238\u7238\u4e70\u793c\u7269"));
    }
}
