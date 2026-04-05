package com.JJLin.aiagent.capability;

import lombok.Builder;
import lombok.Value;

import java.util.List;

@Value
@Builder
public class CapabilityParameter {

    String name;

    String type;

    boolean required;

    String description;

    List<String> aliases;
}
