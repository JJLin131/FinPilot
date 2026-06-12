package com.JJLin.aiagent.api;

import jakarta.validation.Validation;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class FinanceChatRequestValidationTest {

    @Test
    void rejectsMissingTenantId() {
        FinanceChatRequest request = new FinanceChatRequest();
        request.setUserId("u-1");
        request.setChatId("c-1");
        request.setContent("summary");

        try (var factory = Validation.buildDefaultValidatorFactory()) {
            assertTrue(factory.getValidator().validate(request).stream()
                    .anyMatch(violation -> violation.getPropertyPath().toString().equals("tenantId")));
        }
    }
}
