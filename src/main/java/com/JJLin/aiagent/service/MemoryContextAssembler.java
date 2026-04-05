package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.entites.WorkflowMemory;
import com.JJLin.aiagent.enums.VerificationFailureTag;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class MemoryContextAssembler {

    public String assembleForPlan(
            String currentRequest,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<Message> recentMessages,
            String stageNotes) {
        StringBuilder context = new StringBuilder();
        context.append("Current request: ").append(currentRequest).append("\n");
        appendSection(context, "User profile", renderUserProfile(userProfile));
        appendSection(context, "Resolved facts", workflowMemory == null ? null : join(workflowMemory.getResolvedFacts()));
        appendSection(context, "Open questions", workflowMemory == null ? null : join(workflowMemory.getOpenQuestions()));
        appendSection(context, "Latest plan summary", workflowMemory == null ? null : workflowMemory.getLatestPlanSummary());
        appendSection(context, "Latest execution summary", workflowMemory == null ? null : workflowMemory.getLatestExecutionSummary());
        appendSection(context, "Last failure tag", renderFailureTag(workflowMemory));
        appendSection(context, "Last required fixes", workflowMemory == null ? null : join(workflowMemory.getLastRequiredFixes()));
        appendSection(context, "User long-term memory", join(userLongTermMemories));
        appendSection(context, "Recent messages", transcript(recentMessages));
        appendSection(context, "Stage notes", stageNotes);
        return context.toString().trim();
    }

    public String assembleForVerification(
            String currentRequest,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> systemOperationMemories,
            String verificationInput,
            String stageNotes) {
        StringBuilder context = new StringBuilder();
        context.append("Current request: ").append(currentRequest).append("\n");
        appendSection(context, "User profile", renderUserProfile(userProfile));
        appendSection(context, "Resolved facts", workflowMemory == null ? null : join(workflowMemory.getResolvedFacts()));
        appendSection(context, "Open questions", workflowMemory == null ? null : join(workflowMemory.getOpenQuestions()));
        appendSection(context, "Latest plan summary", workflowMemory == null ? null : workflowMemory.getLatestPlanSummary());
        appendSection(context, "Latest execution summary", workflowMemory == null ? null : workflowMemory.getLatestExecutionSummary());
        appendSection(context, "System operation memory", join(systemOperationMemories));
        appendSection(context, "Verification input", verificationInput);
        appendSection(context, "Stage notes", stageNotes);
        return context.toString().trim();
    }

    public String assembleForExecution(
            String currentRequest,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<Message> recentMessages,
            String executionInput,
            String stageNotes) {
        StringBuilder context = new StringBuilder();
        context.append("Current request: ").append(currentRequest).append("\n");
        appendSection(context, "User profile", renderUserProfile(userProfile));
        appendSection(context, "Resolved facts", workflowMemory == null ? null : join(workflowMemory.getResolvedFacts()));
        appendSection(context, "Open questions", workflowMemory == null ? null : join(workflowMemory.getOpenQuestions()));
        appendSection(context, "Latest plan summary", workflowMemory == null ? null : workflowMemory.getLatestPlanSummary());
        appendSection(context, "Latest execution summary", workflowMemory == null ? null : workflowMemory.getLatestExecutionSummary());
        appendSection(context, "User long-term memory", join(userLongTermMemories));
        appendSection(context, "Recent messages", transcript(recentMessages));
        appendSection(context, "Execution input", executionInput);
        appendSection(context, "Stage notes", stageNotes);
        return context.toString().trim();
    }

    public String assembleForResponse(
            String currentRequest,
            WorkflowMemory workflowMemory,
            UserProfile userProfile,
            List<String> userLongTermMemories,
            List<Message> recentMessages,
            String responseInput,
            String stageNotes) {
        StringBuilder context = new StringBuilder();
        context.append("Current request: ").append(currentRequest).append("\n");
        appendSection(context, "User profile", renderUserProfile(userProfile));
        appendSection(context, "Resolved facts", workflowMemory == null ? null : join(workflowMemory.getResolvedFacts()));
        appendSection(context, "Open questions", workflowMemory == null ? null : join(workflowMemory.getOpenQuestions()));
        appendSection(context, "Latest plan summary", workflowMemory == null ? null : workflowMemory.getLatestPlanSummary());
        appendSection(context, "Latest execution summary", workflowMemory == null ? null : workflowMemory.getLatestExecutionSummary());
        appendSection(context, "User long-term memory", join(userLongTermMemories));
        appendSection(context, "Recent messages", transcript(recentMessages));
        appendSection(context, "Response input", responseInput);
        appendSection(context, "Stage notes", stageNotes);
        return context.toString().trim();
    }

    private String renderUserProfile(UserProfile userProfile) {
        if (userProfile == null) {
            return null;
        }
        List<String> fields = new ArrayList<>();
        fields.add(renderField("age", userProfile.getAge() == null ? null : String.valueOf(userProfile.getAge())));
        fields.add(renderField("occupation", userProfile.getOccupation()));
        fields.add(renderField("education", userProfile.getEducation()));
        fields.add(renderField("incomeRange", userProfile.getIncomeRange()));
        fields.add(renderField("gender", userProfile.getGender()));
        fields.add(renderField("city", userProfile.getCity()));
        fields.add(renderField("maritalStatus", userProfile.getMaritalStatus()));
        fields.add(renderField("notes", userProfile.getNotes()));
        return join(fields);
    }

    private String renderField(String label, String value) {
        return value == null || value.isBlank() ? null : label + "=" + value;
    }

    private String renderFailureTag(WorkflowMemory workflowMemory) {
        if (workflowMemory == null || workflowMemory.getLastFailureTag() == null
                || workflowMemory.getLastFailureTag() == VerificationFailureTag.NONE) {
            return null;
        }
        return workflowMemory.getLastFailureTag().name();
    }

    private void appendSection(StringBuilder builder, String title, String value) {
        if (value != null && !value.isBlank()) {
            builder.append(title).append(": ").append(value).append("\n");
        }
    }

    private String join(List<String> values) {
        if (values == null || values.isEmpty()) {
            return null;
        }
        return values.stream()
                .filter(value -> value != null && !value.isBlank())
                .collect(Collectors.joining("; "));
    }

    private String transcript(List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            return null;
        }
        return messages.stream()
                .map(this::render)
                .collect(Collectors.joining(" | "));
    }

    private String render(Message message) {
        if (message instanceof UserMessage userMessage) {
            return "user: " + userMessage.getText();
        }
        if (message instanceof AssistantMessage assistantMessage) {
            return "assistant: " + assistantMessage.getText();
        }
        return message.getMessageType() + ": " + message;
    }
}
