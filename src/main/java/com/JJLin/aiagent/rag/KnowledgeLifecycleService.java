package com.JJLin.aiagent.rag;

import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;

import java.time.LocalDate;

@Service
public class KnowledgeLifecycleService {

    private final RagKnowledgeService knowledgeService;

    public KnowledgeLifecycleService(RagKnowledgeService knowledgeService) {
        this.knowledgeService = knowledgeService;
    }

    @Scheduled(cron = "${ai.knowledge.expiry-cron:0 0 3 * * *}")
    public int removeExpiredDocuments() {
        return knowledgeService.deleteExpiredDocuments(LocalDate.now());
    }
}
