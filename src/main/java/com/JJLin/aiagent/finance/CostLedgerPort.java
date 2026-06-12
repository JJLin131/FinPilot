package com.JJLin.aiagent.finance;

public interface CostLedgerPort {
    CostSnapshot costs(String tenantId, String period);
}
