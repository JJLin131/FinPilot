package com.JJLin.aiagent.entites;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class ChatRequest {

    @NotBlank
    private String userId;

    @NotBlank
    private String chatId;

    @NotBlank
    private String content;
}
