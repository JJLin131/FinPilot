package com.JJLin.aiagent.api;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class FinanceChatRequest {
    @NotBlank
    private String tenantId;
    @NotBlank
    private String userId;
    @NotBlank
    private String chatId;
    @NotBlank
    private String content;
}
