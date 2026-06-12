package com.JJLin.aiagent.finance;

import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Component
public class InMemoryFinanceLedgerAdapter implements SalesLedgerPort, CostLedgerPort, SettlementLedgerPort {

    private static final Map<String, TenantLedger> DATA = Map.of(
            key("tenant-alpha", "2026-05"), new TenantLedger(
                    new BigDecimal("100000.00"),
                    new BigDecimal("5000.00"),
                    new CostSnapshot(new BigDecimal("52000.00"), new BigDecimal("5000.00"),
                            new BigDecimal("1500.00"), new BigDecimal("3500.00")),
                    List.of(
                            entry("sale-1", "tx-1001", "2026-05-02", "CNY", "1000.00"),
                            entry("sale-2", null, "2026-05-03", "CNY", "500.00"),
                            entry("sale-3", null, "2026-05-03", "CNY", "500.00")),
                    List.of(
                            entry("settlement-1", "tx-1001", "2026-05-04", "CNY", "1000.00"),
                            entry("settlement-2", null, "2026-05-03", "CNY", "500.00"),
                            entry("settlement-3", "tx-missing", "2026-05-05", "CNY", "200.00"))),
            key("tenant-beta", "2026-05"), new TenantLedger(
                    new BigDecimal("30000.00"),
                    new BigDecimal("1000.00"),
                    new CostSnapshot(new BigDecimal("14000.00"), new BigDecimal("1200.00"),
                            new BigDecimal("400.00"), new BigDecimal("900.00")),
                    List.of(entry("beta-sale-1", "beta-tx-1", "2026-05-02", "CNY", "700.00")),
                    List.of(entry("beta-settlement-1", "beta-tx-1", "2026-05-03", "CNY", "700.00"))));

    @Override
    public BigDecimal grossSales(String tenantId, String period) {
        return ledger(tenantId, period).grossSales();
    }

    @Override
    public BigDecimal refunds(String tenantId, String period) {
        return ledger(tenantId, period).refunds();
    }

    @Override
    public List<LedgerEntry> salesEntries(String tenantId, String period) {
        return ledger(tenantId, period).sales();
    }

    @Override
    public CostSnapshot costs(String tenantId, String period) {
        return ledger(tenantId, period).costs();
    }

    @Override
    public List<LedgerEntry> settlementEntries(String tenantId, String period) {
        return ledger(tenantId, period).settlements();
    }

    private TenantLedger ledger(String tenantId, String period) {
        TenantLedger ledger = DATA.get(key(tenantId, period));
        if (ledger == null) {
            throw new IllegalArgumentException("No finance fixture data for tenant and period.");
        }
        return ledger;
    }

    private static String key(String tenantId, String period) {
        return tenantId + ":" + period;
    }

    private static LedgerEntry entry(String id, String transactionId, String date, String currency, String amount) {
        return new LedgerEntry(id, transactionId, LocalDate.parse(date), currency, new BigDecimal(amount));
    }

    private record TenantLedger(
            BigDecimal grossSales,
            BigDecimal refunds,
            CostSnapshot costs,
            List<LedgerEntry> sales,
            List<LedgerEntry> settlements) {
    }
}
