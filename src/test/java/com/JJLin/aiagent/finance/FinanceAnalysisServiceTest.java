package com.JJLin.aiagent.finance;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class FinanceAnalysisServiceTest {

    private FinanceAnalysisService service;

    @BeforeEach
    void setUp() {
        InMemoryFinanceLedgerAdapter adapter = new InMemoryFinanceLedgerAdapter();
        service = new FinanceAnalysisService(adapter, adapter, adapter);
    }

    @Test
    void calculatesFinancialSummaryWithBigDecimal() {
        FinancialSummary summary = service.summary("tenant-alpha", "2026-05");

        assertEquals(new BigDecimal("95000.00"), summary.netRevenue());
        assertEquals(new BigDecimal("62000.00"), summary.totalCost());
        assertEquals(new BigDecimal("33000.00"), summary.grossProfit());
        assertEquals(new BigDecimal("0.3474"), summary.grossMargin());
    }

    @Test
    void reconcilesExactMatchesConflictsAndUnmatchedEntries() {
        ReconciliationResult result = service.reconcile("tenant-alpha", "2026-05");

        assertEquals(1, result.matchedCount());
        assertEquals(1, result.unmatchedCount());
        assertEquals(1, result.conflictCount());
        assertEquals(new BigDecimal("1000.00"), result.matchedAmount());
        assertEquals("settlement-3", result.unmatchedIds().get(0));
        assertEquals("settlement-2", result.conflictIds().get(0));
    }

    @Test
    void isolatesTenantFinanceData() {
        FinancialSummary alpha = service.summary("tenant-alpha", "2026-05");
        FinancialSummary beta = service.summary("tenant-beta", "2026-05");

        assertNotEquals(alpha.grossSales(), beta.grossSales());
        assertEquals("tenant-alpha", alpha.tenantId());
        assertEquals("tenant-beta", beta.tenantId());
    }

    @Test
    void returnsNullMarginWhenNetRevenueIsZero() {
        SalesLedgerPort sales = new SalesLedgerPort() {
            public BigDecimal grossSales(String tenantId, String period) { return BigDecimal.TEN; }
            public BigDecimal refunds(String tenantId, String period) { return BigDecimal.TEN; }
            public java.util.List<LedgerEntry> salesEntries(String tenantId, String period) { return java.util.List.of(); }
        };
        CostLedgerPort costs = (tenantId, period) -> new CostSnapshot(
                BigDecimal.ZERO, BigDecimal.ZERO, BigDecimal.ZERO, BigDecimal.ZERO);
        SettlementLedgerPort settlements = (tenantId, period) -> java.util.List.of();

        FinancialSummary summary = new FinanceAnalysisService(sales, costs, settlements).summary("tenant", "2026-05");

        assertNull(summary.grossMargin());
    }
}
