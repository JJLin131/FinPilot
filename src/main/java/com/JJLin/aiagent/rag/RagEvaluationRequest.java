package com.JJLin.aiagent.rag;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

@Data
public class RagEvaluationRequest {
    @Min(1)
    private int k = 5;
    @Valid
    @NotEmpty
    private List<RagEvaluationCase> cases;
}
