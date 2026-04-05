package com.JJLin.aiagent.entites;

import lombok.Data;

import java.util.List;

@Data
public class UserLongTermMemoryExtractionResult {

    private List<String> memories;

    public boolean isEmpty() {
        return memories == null || memories.stream().noneMatch(value -> value != null && !value.isBlank());
    }
}
