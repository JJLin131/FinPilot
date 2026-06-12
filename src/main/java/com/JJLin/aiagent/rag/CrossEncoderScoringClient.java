package com.JJLin.aiagent.rag;

import java.util.List;

public interface CrossEncoderScoringClient {
    List<Double> score(String query, List<String> texts);
}
