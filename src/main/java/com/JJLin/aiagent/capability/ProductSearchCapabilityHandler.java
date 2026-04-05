package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

@Component
public class ProductSearchCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final EcommerceHttpClient httpClient;

    public ProductSearchCapabilityHandler(EcommerceHttpClient httpClient) {
        this.httpClient = httpClient;
    }

    @Override
    public String actionType() {
        return "SEARCH_PRODUCTS";
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
                .description("Search products by a concrete keyword from the user.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("keyword")
                                .type("string")
                                .required(true)
                                .description("Concrete search keyword, such as running shoes or notebook or '手机'.")
                                .build()))
                .exampleParams(Map.of("keyword", "牛奶"))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        Map<String, Object> normalizedParams = pickKnownParams(actionSpec, Map.of(
                "keyword", List.of("keyword", "query", "q")));
        String keyword = normalizedParams.get("keyword") == null ? null : String.valueOf(normalizedParams.get("keyword"));
        if (keyword == null || keyword.isBlank()) {
            return missingParam(actionType(), targetService(), "keyword");
        }
        try {
            Map<String, Object> response = httpClient.get("/api/products/search", normalizedParams);
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("method", "GET");
            evidence.put("path", "/api/products/search");
            evidence.put("normalizedParams", normalizedParams);
            return success(actionType(), targetService(), response, evidence);
        } catch (Exception ex) {
            return failure(actionType(), targetService(), ex, true);
        }
    }
}
