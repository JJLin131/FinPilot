package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KnowledgeChunkerTest {

    @Test
    void splitsByMarkdownHeadingAndIncludesHeadingPath() {
        KnowledgeChunker chunker = chunker(800, 100);
        String content = """
                # 资金规则
                ## 报销审批
                报销金额超过1000元需要主管审批。
                ## 工资发放
                工资发放需要财务经理审批。
                """;

        var chunks = chunker.split(content);

        assertEquals(2, chunks.size());
        assertTrue(chunks.get(0).startsWith("章节路径：资金规则 > 报销审批"));
        assertTrue(chunks.get(1).startsWith("章节路径：资金规则 > 工资发放"));
    }

    @Test
    void removesYamlFrontMatterFromChunks() {
        KnowledgeChunker chunker = chunker(800, 100);
        String content = "---\ndomain: treasury\n---\n# 规则\n正文";

        var chunks = chunker.split(content);

        assertEquals(1, chunks.size());
        assertFalse(chunks.get(0).contains("domain: treasury"));
    }

    @Test
    void keepsMarkdownTableTogether() {
        KnowledgeChunker chunker = chunker(100, 20);
        String table = "| 金额 | 审批人 |\n|---|---|\n" + "| 1000 | 主管 |\n".repeat(20);
        String content = "# 审批规则\n" + table;

        var chunks = chunker.split(content);

        assertEquals(1, chunks.size());
        assertTrue(chunks.get(0).contains("| 1000 | 主管 |"));
    }

    @Test
    void fallsBackToCharacterOverlapForOversizedUnstructuredText() {
        KnowledgeChunker chunker = chunker(100, 20);
        String content = "a".repeat(180);

        var chunks = chunker.split(content);

        assertEquals(2, chunks.size());
        assertEquals(100, chunks.get(0).length());
        assertEquals(100, chunks.get(1).length());
        assertTrue(chunks.get(0).endsWith(chunks.get(1).substring(0, 20)));
    }

    private KnowledgeChunker chunker(int size, int overlap) {
        KnowledgeProperties properties = new KnowledgeProperties();
        properties.setChunkSize(size);
        properties.setChunkOverlap(overlap);
        return new KnowledgeChunker(properties);
    }
}
