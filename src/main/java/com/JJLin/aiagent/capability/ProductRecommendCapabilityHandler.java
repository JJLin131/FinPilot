package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class ProductRecommendCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final EcommerceHttpClient httpClient;

    public ProductRecommendCapabilityHandler(EcommerceHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public String actionType() {
        return "RECOMMEND_PRODUCTS";
    }

    @Override
    public String targetService() {
        return "product-service";
    }

    @Override
    public CapabilityDefinition definition() {
        return CapabilityDefinition.builder()
                .actionType(actionType())
                .targetService(targetService())
                .description("Recommend products using the best available category phrase from the user request, such as running shoes, office supplies, or phones.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("category")
                                .type("string")
                                .required(true)
                                .description("Product category phrase for recommendation. Use the user-stated category directly when available, such as running shoes.")
                                .aliases(List.of("itemCategory", "itemCatagoty"))
                                .build(),
                        CapabilityParameter.builder()
                                .name("limit")
                                .type("integer")
                                .required(false)
                                .description("Maximum number of recommendations to return.")
                                .aliases(List.of("size"))
                                .build()))
                .exampleParams(Map.of("category", "running shoes", "limit", 5))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "category", List.of("category", "itemCategory", "itemCatagoty"),
                "limit", List.of("limit", "size")));
        Object category = normalizedParams.get("category");
        if (category == null || String.valueOf(category).isBlank()) {
            return missingParam(actionType(), targetService(), "category");
        }
        Map<String, Object> backendBody = new LinkedHashMap<>();
        backendBody.put("itemCatagoty", category);
        if (normalizedParams.containsKey("limit")) {
            backendBody.put("limit", normalizedParams.get("limit"));
        }
        try {
            Map<String, Object> response = httpClient.post("/api/products/recommend", backendBody);
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("method", "POST");
            evidence.put("path", "/api/products/recommend");
            evidence.put("normalizedParams", normalizedParams);
            evidence.put("backendBody", backendBody);
            return success(actionType(), targetService(), response, evidence);
        } catch (Exception ex) {
            return failure(actionType(), targetService(), ex, true);
        }
    }
}
