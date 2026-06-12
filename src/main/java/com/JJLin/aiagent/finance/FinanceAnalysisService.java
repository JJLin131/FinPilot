package com.JJLin.aiagent.finance;

import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.List;

@Service
public class FinanceAnalysisService {

    private final SalesLedgerPort salesLedger;
    private final CostLedgerPort costLedger;
    private final SettlementLedgerPort settlementLedger;

    public FinanceAnalysisService(
            SalesLedgerPort salesLedger,
            CostLedgerPort costLedger,
            SettlementLedgerPort settlementLedger) {
        this.salesLedger = salesLedger;
        this.costLedger = costLedger;
        this.settlementLedger = settlementLedger;
    }

    public FinancialSummary summary(String tenantId, String period) {
        BigDecimal grossSales = salesLedger.grossSales(tenantId, period);
        BigDecimal refunds = salesLedger.refunds(tenantId, period);
        BigDecimal netRevenue = grossSales.subtract(refunds);
        BigDecimal totalCost = costLedger.costs(tenantId, period).total();
        BigDecimal grossProfit = netRevenue.subtract(totalCost);
        BigDecimal grossMargin = netRevenue.signum() == 0
                ? null
                : grossProfit.divide(netRevenue, 4, RoundingMode.HALF_UP);
        return new FinancialSummary(tenantId, period, grossSales, refunds, netRevenue, totalCost, grossProfit, grossMargin);
    }

    public CostSnapshot costs(String tenantId, String period) {
        return costLedger.costs(tenantId, period);
    }

    public ReconciliationResult reconcile(String tenantId, String period) {
        List<LedgerEntry> sales = salesLedger.salesEntries(tenantId, period);
        List<String> unmatched = new ArrayList<>();
        List<String> conflicts = new ArrayList<>();
        int matched = 0;
        BigDecimal matchedAmount = BigDecimal.ZERO;

        for (LedgerEntry settlement : settlementLedger.settlementEntries(tenantId, period)) {
            List<LedgerEntry> candidates = candidates(settlement, sales);
            if (candidates.size() == 1) {
                matched++;
                matchedAmount = matchedAmount.add(settlement.amount());
            } else if (candidates.isEmpty()) {
                unmatched.add(settlement.id());
            } else {
                conflicts.add(settlement.id());
            }
        }
        return new ReconciliationResult(tenantId, period, matched, unmatched.size(), conflicts.size(),
                matchedAmount, List.copyOf(unmatched), List.copyOf(conflicts));
    }

    private List<LedgerEntry> candidates(LedgerEntry settlement, List<LedgerEntry> sales) {
        if (settlement.transactionId() != null && !settlement.transactionId().isBlank()) {
            return sales.stream()
                    .filter(sale -> settlement.transactionId().equals(sale.transactionId()))
                    .toList();
        }
        return sales.stream()
                .filter(sale -> settlement.currency().equals(sale.currency()))
                .filter(sale -> settlement.amount().compareTo(sale.amount()) == 0)
                .filter(sale -> settlement.businessDate().equals(sale.businessDate()))
                .toList();
    }
}
