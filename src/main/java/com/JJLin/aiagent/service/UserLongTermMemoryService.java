package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.ChatRequest;
import com.JJLin.aiagent.entites.UserProfile;
import com.JJLin.aiagent.extractor.UserLongTermMemoryExtractorService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class UserLongTermMemoryService {

    private static final double SIMILARITY_THRESHOLD = 0.2d;
    private static final Logger log = LoggerFactory.getLogger(UserLongTermMemoryService.class);

    private final VectorStore userLongTermVectorStore;
    private final UserLongTermMemoryExtractorService extractorService;
    private final Map<String, Set<String>> memoryIndex = new ConcurrentHashMap<>();

    public UserLongTermMemoryService(
            VectorStore userLongTermVectorStore,
            UserLongTermMemoryExtractorService extractorService) {
        this.userLongTermVectorStore = userLongTermVectorStore;
        this.extractorService = extractorService;
    }

    @Async("memoryTaskExecutor")
    public void extractFromRequestAsync(ChatRequest request, UserProfile existingProfile) {
        try {
            List<String> memories = extractorService.extract(request.getUserId(), request.getContent(), existingProfile);
            if (!memories.isEmpty()) {
                saveAll(request.getUserId(), memories);
            }
        } catch (Exception ex) {
            log.warn("user long term memory async update failed. userId={}", request.getUserId(), ex);
        }
    }

    public void saveAll(String userId, List<String> contents) {
        if (userId == null || userId.isBlank() || contents == null || contents.isEmpty()) {
            return;
        }
        Set<String> existing = memoryIndex.computeIfAbsent(userId, ignored -> ConcurrentHashMap.newKeySet());
        List<Document> documents = contents.stream()
                .filter(content -> content != null && !content.isBlank())
                .map(String::trim)
                .filter(content -> existing.add(content))
                .map(content -> Document.builder()
                        .id(userId + ":" + Integer.toUnsignedString(content.hashCode()))
                        .text(content)
                        .metadata(Map.of("userId", userId))
                        .build())
                .toList();
        if (!documents.isEmpty()) {
            userLongTermVectorStore.add(documents);
        }
    }

    public List<String> findRelevant(String userId, String query, int limit) {
        if (userId == null || userId.isBlank() || query == null || query.isBlank()) {
            return List.of();
        }
        List<Document> documents = userLongTermVectorStore.similaritySearch(SearchRequest.builder()
                .query(query)
                .topK(Math.max(limit * 3, 10))
                .similarityThreshold(SIMILARITY_THRESHOLD)
                .build());
        return documents.stream()
                .filter(document -> userId.equals(document.getMetadata().get("userId")))
                .map(Document::getText)
                .filter(text -> text != null && !text.isBlank())
                .distinct()
                .limit(Math.max(1, limit))
                .toList();
    }
}
