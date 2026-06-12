package com.JJLin.aiagent.finance;

import java.math.BigDecimal;
import java.util.List;

public interface SalesLedgerPort {
    BigDecimal grossSales(String tenantId, String period);
    BigDecimal refunds(String tenantId, String period);
    List<LedgerEntry> salesEntries(String tenantId, String period);
}
