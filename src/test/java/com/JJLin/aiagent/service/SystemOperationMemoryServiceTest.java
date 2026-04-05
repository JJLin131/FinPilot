package com.JJLin.aiagent.service;

import com.JJLin.aiagent.entites.SystemOperationMemory;
import com.JJLin.aiagent.mapper.SystemOperationMemoryMapper;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class SystemOperationMemoryServiceTest {

    @Test
    void shouldTruncateLongContentBeforeInsert() {
        SystemOperationMemoryMapper mapper = mock(SystemOperationMemoryMapper.class);
        SystemOperationMemoryService service = new SystemOperationMemoryService(mapper);

        service.record("u-1", "CHECK_ORDER", "x".repeat(800));

        ArgumentCaptor<SystemOperationMemory> captor = ArgumentCaptor.forClass(SystemOperationMemory.class);
        verify(mapper).insert(captor.capture());

        SystemOperationMemory memory = captor.getValue();
        assertNotNull(memory);
        assertNotNull(memory.getContent());
        assertTrue(memory.getContent().length() <= 500);
        assertTrue(memory.getContent().endsWith("...[truncated]"));
    }
}
