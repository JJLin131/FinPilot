package com.JJLin.aiagent.finance;

import java.math.BigDecimal;

public record FinancialSummary(
        String tenantId,
        String period,
        BigDecimal grossSales,
        BigDecimal refunds,
        BigDecimal netRevenue,
        BigDecimal totalCost,
        BigDecimal grossProfit,
        BigDecimal grossMargin) {
}
