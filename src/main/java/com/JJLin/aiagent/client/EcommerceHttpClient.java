package com.JJLin.aiagent.client;

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.util.UriBuilder;

import java.net.URI;
import java.util.Map;

@Component
public class EcommerceHttpClient {

    private final RestClient restClient;

    public EcommerceHttpClient(RestClient restClient) {
        this.restClient = restClient;
    }

    public Map<String, Object> get(String path, Map<String, Object> queryParams) {
        return restClient.get()
                .uri(uriBuilder -> buildUri(uriBuilder, path, queryParams))
                .accept(MediaType.APPLICATION_JSON)
                .retrieve()
                .body(new ParameterizedTypeReference<>() {});
    }

    public Map<String, Object> post(String path, Object body) {
        return restClient.post()
                .uri(path)
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(new ParameterizedTypeReference<>() {});
    }

    private URI buildUri(UriBuilder uriBuilder, String path, Map<String, Object> queryParams) {
        uriBuilder.path(path);
        if (queryParams != null) {
            queryParams.forEach(uriBuilder::queryParam);
        }
        return uriBuilder.build();
    }
}
