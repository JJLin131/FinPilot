package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

public class OrderCreateCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final EcommerceHttpClient httpClient;

    public OrderCreateCapabilityHandler(EcommerceHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public String actionType() {
        return "CREATE_ORDER";
    }

    @Override
    public String targetService() {
        return "order-service";
    }

    @Override
    public CapabilityDefinition definition() {
        return CapabilityDefinition.builder()
                .actionType(actionType())
                .targetService(targetService())
                .description("Create an order for a concrete user account and sku.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("userId")
                                .type("string")
                                .required(true)
                                .description("User identifier.")
                                .aliases(List.of())
                                .build(),
                        CapabilityParameter.builder()
                                .name("accountId")
                                .type("string")
                                .required(true)
                                .description("Account identifier for checkout.")
                                .aliases(List.of())
                                .build(),
                        CapabilityParameter.builder()
                                .name("skuId")
                                .type("string")
                                .required(true)
                                .description("SKU identifier to purchase.")
                                .aliases(List.of("productSkuId"))
                                .build()))
                .exampleParams(Map.of("userId", "u-1", "accountId", "a-1", "skuId", "sku-100"))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "userId", List.of("userId"),
                "accountId", List.of("accountId"),
                "skuId", List.of("skuId", "productSkuId")));
        String userId = normalizedParams.get("userId") == null ? null : String.valueOf(normalizedParams.get("userId"));
        String accountId = normalizedParams.get("accountId") == null ? null : String.valueOf(normalizedParams.get("accountId"));
        String skuId = normalizedParams.get("skuId") == null ? null : String.valueOf(normalizedParams.get("skuId"));
        if (userId == null || accountId == null || skuId == null) {
            return missingParam(actionType(), targetService(), "userId/accountId/skuId");
        }
        try {
            Map<String, Object> response = httpClient.post("/api/orders/create", normalizedParams);
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("method", "POST");
            evidence.put("path", "/api/orders/create");
            evidence.put("normalizedParams", normalizedParams);
            return success(actionType(), targetService(), response, evidence);
        } catch (Exception ex) {
            return failure(actionType(), targetService(), ex, false);
        }
    }
}
