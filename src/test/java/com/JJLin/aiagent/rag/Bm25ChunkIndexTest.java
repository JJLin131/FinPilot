package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Bm25ChunkIndexTest {

    @TempDir
    Path tempDir;

    @Test
    void retrievesChineseExactTermsWithBm25() {
        KnowledgeProperties properties = new KnowledgeProperties();
        properties.setBm25IndexPath(tempDir.toString());
        Bm25ChunkIndex index = new Bm25ChunkIndex(properties);
        KnowledgeDocumentRequest request = new KnowledgeDocumentRequest();
        request.setDocumentId("doc-1");
        request.setDomain(KnowledgeDomain.FINANCE);
        request.setTitle("工资审批规则");
        request.setSource("salary.md");
        request.setTags(List.of("工资", "审批"));
        index.replaceDocument(request, List.of("chunk-1", "chunk-2"), List.of(
                "工资发放超过五十万元需要财务总监审批。",
                "普通报销需要部门负责人审批。"));

        var results = index.search(KnowledgeDomain.FINANCE, "工资发放", 5);

        assertEquals(2, results.size());
        assertEquals("工资发放超过五十万元需要财务总监审批。", results.get(0).text());
    }
}
