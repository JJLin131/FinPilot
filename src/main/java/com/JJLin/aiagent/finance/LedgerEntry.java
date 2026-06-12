package com.JJLin.aiagent.finance;

import java.math.BigDecimal;
import java.time.LocalDate;

public record LedgerEntry(String id, String transactionId, LocalDate businessDate, String currency, BigDecimal amount) {
}
