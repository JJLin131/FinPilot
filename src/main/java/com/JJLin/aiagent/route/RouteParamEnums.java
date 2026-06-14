package com.JJLin.aiagent.route;

public enum RouteParamEnums {
    CANCEL_TRANSFER,
    TRANSFER_STATUS,
    TRANSFER_RECEIPT,
    ACCOUNT_BALANCE,
    KNOWLEDGE_QA,
    GENERAL_FINANCE,
    UNKNOWN;

    public static RouteParamEnums fromValue(String value) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("intent must not be blank");
        }
        return RouteParamEnums.valueOf(value.trim().toUpperCase());
    }
}
