package com.JJLin.aiagent.controller;

import com.JJLin.aiagent.rag.KnowledgeDocumentRequest;
import com.JJLin.aiagent.rag.KnowledgeDocumentResult;
import com.JJLin.aiagent.rag.KnowledgeLifecycleService;
import com.JJLin.aiagent.rag.RagKnowledgeService;
import com.JJLin.aiagent.rag.RagEvaluationRequest;
import com.JJLin.aiagent.rag.RagEvaluationResult;
import com.JJLin.aiagent.rag.RagEvaluationService;
import com.JJLin.aiagent.rag.RagCurationAgent;
import com.JJLin.aiagent.rag.ResourceKnowledgeImporter;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;
import java.util.List;

@RestController
@RequestMapping("/api/knowledge")
public class KnowledgeAdminController {

    private final RagKnowledgeService knowledgeService;
    private final KnowledgeLifecycleService lifecycleService;
    private final RagEvaluationService evaluationService;
    private final RagCurationAgent curationAgent;
    private final ResourceKnowledgeImporter resourceImporter;

    public KnowledgeAdminController(
            RagKnowledgeService knowledgeService,
            KnowledgeLifecycleService lifecycleService,
            RagEvaluationService evaluationService,
            RagCurationAgent curationAgent,
            ResourceKnowledgeImporter resourceImporter) {
        this.knowledgeService = knowledgeService;
        this.lifecycleService = lifecycleService;
        this.evaluationService = evaluationService;
        this.curationAgent = curationAgent;
        this.resourceImporter = resourceImporter;
    }

    @PostMapping("/documents")
    public KnowledgeDocumentResult ingest(@Valid @RequestBody KnowledgeDocumentRequest request) {
        return knowledgeService.ingest(curationAgent.curate(request));
    }

    @DeleteMapping("/expired")
    public Map<String, Integer> deleteExpired() {
        return Map.of("deletedDocuments", lifecycleService.removeExpiredDocuments());
    }

    @PostMapping("/evaluate")
    public RagEvaluationResult evaluate(@Valid @RequestBody RagEvaluationRequest request) {
        return evaluationService.evaluate(request);
    }

    @PostMapping("/bootstrap/resources")
    public List<KnowledgeDocumentResult> importResources() {
        return resourceImporter.importFinanceResources();
    }
}
