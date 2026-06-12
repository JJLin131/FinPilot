package com.JJLin.aiagent.tools;

import com.JJLin.aiagent.agent.AgentRequestContext;
import com.JJLin.aiagent.api.AgentEvidence;
import com.JJLin.aiagent.finance.CostSnapshot;
import com.JJLin.aiagent.finance.FinanceAnalysisService;
import com.JJLin.aiagent.finance.FinancialSummary;
import com.JJLin.aiagent.finance.ReconciliationResult;
import com.JJLin.aiagent.service.AgentToolAuditService;
import com.JJLin.aiagent.rag.KnowledgeDomain;
import com.JJLin.aiagent.rag.RagKnowledgeService;
import com.JJLin.aiagent.rag.RagMatch;
import dev.langchain4j.agent.tool.P;
import dev.langchain4j.agent.tool.Tool;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.List;
import java.util.function.Supplier;

@Component
public class FinanceTools {

    private final FinanceAnalysisService analysisService;
    private final AgentToolAuditService auditService;
    private final RagKnowledgeService knowledgeService;

    public FinanceTools(
            FinanceAnalysisService analysisService,
            AgentToolAuditService auditService,
            RagKnowledgeService knowledgeService) {
        this.analysisService = analysisService;
        this.auditService = auditService;
        this.knowledgeService = knowledgeService;
    }

    @Tool("Get authoritative financial summary for a YYYY-MM period.")
    public FinancialSummary getFinancialSummary(@P("Period formatted as YYYY-MM") String period) {
        return execute("getFinancialSummary", period, () -> analysisService.summary(tenantId(), period));
    }

    @Tool("Calculate authoritative costs for a YYYY-MM period. Scenario is explanatory only and never changes source data.")
    public CostSnapshot calculateCost(
            @P("Period formatted as YYYY-MM") String period,
            @P("Optional scenario description") String scenario) {
        return execute("calculateCost", period, () -> analysisService.costs(tenantId(), period));
    }

    @Tool("Reconcile settlement transactions against sales using deterministic rules.")
    public ReconciliationResult reconcileTransactions(@P("Period formatted as YYYY-MM") String period) {
        return execute("reconcileTransactions", period, () -> analysisService.reconcile(tenantId(), period));
    }

    @Tool("Explain an authoritative financial metric using the financial summary.")
    public Map<String, Object> explainFinancialMetric(
            @P("Metric name") String metricName,
            @P("Period formatted as YYYY-MM") String period) {
        FinancialSummary summary = execute("explainFinancialMetric", period, () -> analysisService.summary(tenantId(), period));
        return Map.of("metricName", metricName, "period", period, "financialSummary", summary);
    }

    @Tool("Ask the tenant-scoped finance RAG sub-agent about policies, contracts, regulations, or accounting knowledge.")
    public List<RagMatch> askFinanceKnowledge(@P("Finance knowledge question") String question) {
        return knowledgeService.search(KnowledgeDomain.FINANCE, tenantId(), question, 5);
    }

    private String tenantId() {
        String tenantId = AgentRequestContext.require().tenantId();
        if (tenantId == null || tenantId.isBlank()) {
            throw new IllegalStateException("Finance tools require a trusted tenant context.");
        }
        return tenantId;
    }

    private <T> T execute(String tool, String period, Supplier<T> action) {
        long start = System.nanoTime();
        try {
            T result = action.get();
            AgentRequestContext.addEvidence(AgentEvidence.builder()
                    .toolName(tool).period(period).source("finance-fixture-ledger")
                    .summary(Map.of("authoritative", true)).build());
            auditService.record(tool, "period=" + period, "SUCCEEDED", elapsedMs(start));
            return result;
        } catch (RuntimeException exception) {
            auditService.record(tool, "period=" + period, "FAILED", elapsedMs(start));
            throw exception;
        }
    }

    private long elapsedMs(long start) {
        return (System.nanoTime() - start) / 1_000_000;
    }
}
