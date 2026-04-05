package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.WorkflowMemory;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@Service
public class MemoryCompressionService {

    private static final int MAX_SUMMARY_LENGTH = 1000;

    public String compress(WorkflowMemory workflowMemory) {
        List<String> sections = new ArrayList<>();
        appendIfPresent(sections, "Resolved facts", join(workflowMemory.getResolvedFacts()));
        appendIfPresent(sections, "Open questions", join(workflowMemory.getOpenQuestions()));
        appendIfPresent(sections, "Latest plan", workflowMemory.getLatestPlanSummary());
        appendIfPresent(sections, "Latest execution", workflowMemory.getLatestExecutionSummary());
        appendIfPresent(sections, "Last failure", workflowMemory.getLastFailureTag() == null ? null : workflowMemory.getLastFailureTag().name());
        appendIfPresent(sections, "Required fixes", join(workflowMemory.getLastRequiredFixes()));

        String summary = String.join("\n", sections).trim();
        if (summary.length() <= MAX_SUMMARY_LENGTH) {
            return summary;
        }
        return summary.substring(summary.length() - MAX_SUMMARY_LENGTH);
    }

    private void appendIfPresent(List<String> sections, String title, String content) {
        if (content != null && !content.isBlank()) {
            sections.add(title + ": " + content);
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
}