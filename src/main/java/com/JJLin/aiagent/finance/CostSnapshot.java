package com.JJLin.aiagent.finance;

import java.math.BigDecimal;

public record CostSnapshot(BigDecimal cogs, BigDecimal platformFees, BigDecimal paymentFees, BigDecimal shippingCost) {
    public BigDecimal total() {
        return cogs.add(platformFees).add(paymentFees).add(shippingCost);
    }
}
