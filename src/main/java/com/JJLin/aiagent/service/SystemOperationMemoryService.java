package com.JJLin.aiagent.service;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.JJLin.aiagent.entites.SystemOperationMemory;
import com.JJLin.aiagent.mapper.SystemOperationMemoryMapper;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class SystemOperationMemoryService {

    private static final int MAX_CONTENT_LENGTH = 500;
    private static final String TRUNCATED_SUFFIX = " ...[truncated]";

    private final SystemOperationMemoryMapper systemOperationMemoryMapper;

    public SystemOperationMemoryService(SystemOperationMemoryMapper systemOperationMemoryMapper) {
        this.systemOperationMemoryMapper = systemOperationMemoryMapper;
    }

    public void record(String userId, String operationType, String content) {
        if (userId == null || userId.isBlank() || content == null || content.isBlank()) {
            return;
        }
        SystemOperationMemory memory = new SystemOperationMemory();
        memory.setUserId(userId);
        memory.setOperationType(operationType == null || operationType.isBlank() ? "GENERAL" : operationType);
        memory.setContent(normalizeContent(content));
        memory.setCreatedAt(LocalDateTime.now());
        systemOperationMemoryMapper.insert(memory);
    }

    public List<String> findRelevant(String userId, String operationType, int limit) {
        if (userId == null || userId.isBlank()) {
            return List.of();
        }
        LambdaQueryWrapper<SystemOperationMemory> wrapper = new LambdaQueryWrapper<SystemOperationMemory>()
                .eq(SystemOperationMemory::getUserId, userId)
                .orderByDesc(SystemOperationMemory::getCreatedAt)
                .last("limit " + Math.max(1, limit));
        if (operationType != null && !operationType.isBlank()) {
            wrapper.eq(SystemOperationMemory::getOperationType, operationType);
        }
        return systemOperationMemoryMapper.selectList(wrapper).stream()
                .map(SystemOperationMemory::getContent)
                .toList();
    }

    private String normalizeContent(String content) {
        String normalized = content.trim();
        if (normalized.length() <= MAX_CONTENT_LENGTH) {
            return normalized;
        }
        int endIndex = Math.max(0, MAX_CONTENT_LENGTH - TRUNCATED_SUFFIX.length());
        return normalized.substring(0, endIndex) + TRUNCATED_SUFFIX;
    }
}
