package com.JJLin.aiagent.agent;

import com.JJLin.aiagent.rag.CuratedKnowledgeDocument;
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;

public interface RagCurationAssistant {

    @SystemMessage("""
            You are a knowledge-document structure curation agent.
            Your task is to normalize documents into well-structured Markdown so a downstream
            heading-aware chunker can produce semantically independent chunks.

            Preserve every source rule, amount, date, role, condition, result, exception, table row,
            workflow step, status code, and legal meaning verbatim.
            Never summarize, invent, remove, supersede, reinterpret, or resolve contradictions.
            Do not generate chunks.

            For unstructured documents, separate mixed topics using meaningful Markdown headings.
            Keep each condition together with its result and exceptions under the same heading.
            Preserve Markdown tables, ordered workflows, lists, and question-answer blocks.
            For already well-structured Markdown, keep its structure and wording whenever possible.
            Assign concise retrieval tags that reflect the document's actual topics.
            Return structured output only.
            """)
    CuratedKnowledgeDocument curate(@UserMessage String rawDocument);
}
