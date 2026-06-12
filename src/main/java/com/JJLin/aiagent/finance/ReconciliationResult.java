package com.JJLin.aiagent.finance;

import java.math.BigDecimal;
import java.util.List;

public record ReconciliationResult(
        String tenantId,
        String period,
        int matchedCount,
        int unmatchedCount,
        int conflictCount,
        BigDecimal matchedAmount,
        List<String> unmatchedIds,
        List<String> conflictIds) {
}
