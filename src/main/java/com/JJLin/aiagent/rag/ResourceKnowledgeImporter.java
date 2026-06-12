package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

@Service
public class ResourceKnowledgeImporter {

    public static final String GLOBAL_TENANT = "__GLOBAL__";

    private final KnowledgeProperties properties;
    private final RagCurationAgent curationAgent;
    private final RagKnowledgeService knowledgeService;

    public ResourceKnowledgeImporter(
            KnowledgeProperties properties,
            RagCurationAgent curationAgent,
            RagKnowledgeService knowledgeService) {
        this.properties = properties;
        this.curationAgent = curationAgent;
        this.knowledgeService = knowledgeService;
    }

    public List<KnowledgeDocumentResult> importFinanceResources() {
        try {
            Resource[] resources = new PathMatchingResourcePatternResolver()
                    .getResources(properties.getResourcePattern());
            List<KnowledgeDocumentResult> results = new ArrayList<>();
            for (Resource resource : resources) {
                if (!resource.isReadable() || !isFinanceResource(resource)) {
                    continue;
                }
                KnowledgeDocumentRequest request = new KnowledgeDocumentRequest();
                request.setDocumentId(documentId(resource));
                request.setDomain(KnowledgeDomain.FINANCE);
                request.setTenantId(GLOBAL_TENANT);
                request.setTitle(title(resource));
                request.setSource(resource.getDescription());
                request.setContent(resource.getContentAsString(StandardCharsets.UTF_8));
                results.add(knowledgeService.ingest(curationAgent.curate(request)));
            }
            return results;
        } catch (IOException exception) {
            throw new IllegalStateException("Failed to import resource knowledge documents.", exception);
        }
    }

    private boolean isFinanceResource(Resource resource) {
        String filename = resource.getFilename();
        return filename != null && filename.matches("\\d{2}_.+\\.md");
    }

    private String documentId(Resource resource) {
        String filename = resource.getFilename();
        return "resource-finance-" + Integer.toUnsignedString(
                (filename == null ? resource.getDescription() : filename).hashCode());
    }

    private String title(Resource resource) {
        String filename = resource.getFilename();
        if (filename == null) {
            return resource.getDescription();
        }
        return filename.endsWith(".md") ? filename.substring(0, filename.length() - 3) : filename;
    }
}
