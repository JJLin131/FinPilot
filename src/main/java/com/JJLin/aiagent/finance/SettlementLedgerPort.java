package com.JJLin.aiagent.finance;

import java.util.List;

public interface SettlementLedgerPort {
    List<LedgerEntry> settlementEntries(String tenantId, String period);
}
