package com.JJLin.aiagent.rag;

import com.JJLin.aiagent.config.KnowledgeProperties;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.http.client.SimpleClientHttpRequestFactory;

import java.util.Comparator;
import java.util.List;
import java.util.Map;

@Component
public class HttpCrossEncoderScoringClient implements CrossEncoderScoringClient {

    private final RestClient restClient;

    public HttpCrossEncoderScoringClient(KnowledgeProperties properties) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(properties.getRerankerTimeoutSeconds() * 1000);
        requestFactory.setReadTimeout(properties.getRerankerTimeoutSeconds() * 1000);
        this.restClient = RestClient.builder()
                .baseUrl(properties.getRerankerBaseUrl())
                .requestFactory(requestFactory)
                .build();
    }

    @Override
    public List<Double> score(String query, List<String> texts) {
        List<RerankResponse> response = restClient.post()
                .uri("/rerank")
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of(
                        "query", query,
                        "texts", texts,
                        "raw_scores", false,
                        "return_text", false,
                        "truncate", true))
                .retrieve()
                .body(new ParameterizedTypeReference<>() {});
        if (response == null || response.size() != texts.size()) {
            throw new IllegalStateException("Cross-encoder returned an invalid response.");
        }
        return response.stream()
                .sorted(Comparator.comparingInt(RerankResponse::index))
                .map(RerankResponse::score)
                .toList();
    }

    private record RerankResponse(int index, double score) {
    }
}
