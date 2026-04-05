package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class OrderQueryCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final EcommerceHttpClient httpClient;

    public OrderQueryCapabilityHandler(EcommerceHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public String actionType() {
        return "CHECK_ORDER";
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
                .description("Query a single order by orderId using the backend path /order/{orderId}.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("orderId")
                                .type("string")
                                .required(true)
                                .description("Order identifier used in the path segment /order/{orderId}.")
                                .aliases(List.of("orderNo"))
                                .build()))
                .exampleParams(Map.of("orderId", "123456"))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "orderId", List.of("orderId", "orderNo")));
        String orderId = normalizedParams.get("orderId") == null ? null : String.valueOf(normalizedParams.get("orderId"));
        if (orderId == null || orderId.isBlank()) {
            return missingParam(actionType(), targetService(), "orderId");
        }
        try {
            String path = "/order/" + URLEncoder.encode(orderId, StandardCharsets.UTF_8);
            Map<String, Object> response = httpClient.get(path, Map.of());
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("method", "GET");
            evidence.put("path", path);
            evidence.put("normalizedParams", normalizedParams);
            return success(actionType(), targetService(), response, evidence);
        } catch (Exception ex) {
            return failure(actionType(), targetService(), ex, true);
        }
    }
}
