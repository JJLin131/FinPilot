package com.JJLin.aiagent.entites;

import com.JJLin.aiagent.enums.VerificationFailureTag;
import com.JJLin.aiagent.enums.VerificationType;
import lombok.Data;

import java.util.List;

@Data
public class VerificationResult {

    private Boolean pass;

    private VerificationType verificationType;

    private List<String> findings;

    private List<String> requiredFixes;

    private VerificationFailureTag failureTag;

    private String summary;
}
