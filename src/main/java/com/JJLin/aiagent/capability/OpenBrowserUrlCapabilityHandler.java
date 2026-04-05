package com.JJLin.aiagent.capability;

import com.JJLin.aiagent.client.BrowserClient;
import com.JJLin.aiagent.client.BrowserOperationException;
import com.JJLin.aiagent.config.CapabilityProperties;
import com.JJLin.aiagent.entites.ActionSpec;
import com.JJLin.aiagent.entites.ExecutionResult;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Component
public class OpenBrowserUrlCapabilityHandler extends AbstractHttpCapabilityHandler {

    private final BrowserClient browserClient;
    private final CapabilityProperties properties;

    public OpenBrowserUrlCapabilityHandler(BrowserClient browserClient, CapabilityProperties properties) {
        this.browserClient = browserClient;
        this.properties = properties;
    }

    @Override
    public String actionType() {
        return "OPEN_BROWSER_URL";
    }

    @Override
    public String targetService() {
        return "system-browser";
    }

    @Override
    public CapabilityDefinition definition() {
        return CapabilityDefinition.builder()
                .actionType(actionType())
                .targetService(targetService())
                .description("Open a URL in the system browser.")
                .parameters(List.of(
                        CapabilityParameter.builder()
                                .name("url")
                                .type("string")
                                .required(true)
                                .description("HTTP or HTTPS URL to open.")
                                .aliases(List.of())
                                .build()))
                .exampleParams(Map.of("url", "https://baidu.com"))
                .build();
    }

    @Override
    public ExecutionResult execute(ActionSpec actionSpec) {
        String url = stringParam(actionSpec, "url");
        if (url == null || url.isBlank()) {
            return missingParam(actionType(), targetService(), "url");
        }
        try {
            URI uri = URI.create(url);
            if (uri.getScheme() == null || properties.getBrowserAllowedSchemes().stream()
                    .noneMatch(allowed -> allowed.equalsIgnoreCase(uri.getScheme()))) {
                return invalidParam(actionType(), targetService(), "Unsupported URL scheme: " + uri.getScheme());
            }
            browserClient.open(uri);
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("url", uri.toString());
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("allowedSchemes", properties.getBrowserAllowedSchemes());
            return success(actionType(), targetService(), payload, evidence);
        } catch (IllegalArgumentException ex) {
            return invalidParam(actionType(), targetService(), ex.getMessage());
        } catch (BrowserOperationException ex) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("exceptionType", ex.getClass().getName());
            evidence.put("url", url);
            evidence.put("allowedSchemes", properties.getBrowserAllowedSchemes());
            return failure(actionType(), targetService(), ex.getErrorCode(), ex.getMessage(), evidence, false);
        } catch (Exception ex) {
            Map<String, Object> evidence = new LinkedHashMap<>();
            evidence.put("exceptionType", ex.getClass().getName());
            evidence.put("url", url);
            evidence.put("allowedSchemes", properties.getBrowserAllowedSchemes());
            return failure(actionType(), targetService(), "BROWSER_OPEN_FAILED", ex.getMessage(), evidence, false);
        }
    }
}
