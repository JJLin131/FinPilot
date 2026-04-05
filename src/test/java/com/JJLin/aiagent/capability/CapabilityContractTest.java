package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import com.JJLin.aiagent.enums.WorkflowTaskStatus;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CapabilityContractTest {

    @Test
    void shouldRejectUnsupportedParamsAgainstCapabilitySchema() {
        CapabilityRegistry registry = new CapabilityRegistry(List.of(
                new ProductSearchCapabilityHandler(mock(EcommerceHttpClient.class))));

        List<String> issues = registry.validateAction(
                "SEARCH_PRODUCTS",
                "product-service",
                Map.of("userRole", "teacher"));

        assertEquals(2, issues.size());
        assertTrue(issues.stream().anyMatch(issue -> issue.contains("Missing required param 'keyword'")));
        assertTrue(issues.stream().anyMatch(issue -> issue.contains("Unsupported param 'userRole'")));
    }

    @Test
    void shouldNormalizeRecommendationCategoryBeforeCallingBackend() {
        EcommerceHttpClient httpClient = mock(EcommerceHttpClient.class);
        ProductRecommendCapabilityHandler handler = new ProductRecommendCapabilityHandler(httpClient);
        when(httpClient.post(eq("/api/products/recommend"), any()))
                .thenReturn(Map.of("items", List.of("marker")));

        ActionSpec actionSpec = new ActionSpec();
        actionSpec.setActionType("RECOMMEND_PRODUCTS");
        actionSpec.setTargetService("product-service");
        actionSpec.setParams(Map.of("category", "office supplies", "limit", 3));

        ExecutionResult result = handler.execute(actionSpec);

        assertEquals(WorkflowTaskStatus.SUCCEEDED, result.getStatus());
        verify(httpClient).post("/api/products/recommend", Map.of("itemCatagoty", "office supplies", "limit", 3));
    }
}
