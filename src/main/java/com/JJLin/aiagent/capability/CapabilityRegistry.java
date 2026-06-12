package com.JJLin.aiagent.capability;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

public class CapabilityRegistry {

    private final Map<String, CapabilityHandler> handlers = new LinkedHashMap<>();

    public CapabilityRegistry(List<CapabilityHandler> capabilityHandlers) {
        for (CapabilityHandler handler : capabilityHandlers) {
            handlers.put(handler.actionType(), handler);
        }
    }

    public Optional<CapabilityHandler> find(String actionType) {
        return Optional.ofNullable(handlers.get(actionType));
    }

    public Map<String, Object> describe() {
        Map<String, Object> result = new LinkedHashMap<>();
        handlers.forEach((key, value) -> result.put(key, value.definition()));
        return result;
    }

    public Optional<CapabilityDefinition> definition(String actionType) {
        return find(actionType).map(CapabilityHandler::definition);
    }

    public List<String> validateAction(String actionType, String targetService, Map<String, Object> params) {
        CapabilityHandler handler = handlers.get(actionType);
        if (handler == null) {
            return List.of("Unsupported actionType: " + actionType);
        }

        CapabilityDefinition definition = handler.definition();
        List<String> issues = new ArrayList<>();
        if (targetService != null && !targetService.isBlank() && !definition.getTargetService().equals(targetService)) {
            issues.add("targetService for " + actionType + " must be " + definition.getTargetService() + ".");
        }

        Map<String, Object> safeParams = params == null ? Collections.emptyMap() : params;
        Set<String> allowedNames = definition.getParameters().stream()
                .flatMap(parameter -> {
                    List<String> names = new ArrayList<>();
                    names.add(parameter.getName());
                    if (parameter.getAliases() != null) {
                        names.addAll(parameter.getAliases());
                    }
                    return names.stream();
                })
                .collect(Collectors.toSet());

        definition.getParameters().stream()
                .filter(CapabilityParameter::isRequired)
                .filter(parameter -> parameterNames(parameter).stream().noneMatch(safeParams::containsKey))
                .forEach(parameter -> issues.add("Missing required param '" + parameter.getName()
                        + "' for actionType " + actionType + "."));

        safeParams.keySet().stream()
                .filter(paramName -> !allowedNames.contains(paramName))
                .forEach(paramName -> issues.add("Unsupported param '" + paramName
                        + "' for actionType " + actionType + "."));
        return issues;
    }

    private List<String> parameterNames(CapabilityParameter parameter) {
        List<String> names = new ArrayList<>();
        names.add(parameter.getName());
        if (parameter.getAliases() != null) {
            names.addAll(parameter.getAliases());
        }
        return names;
    }
}
