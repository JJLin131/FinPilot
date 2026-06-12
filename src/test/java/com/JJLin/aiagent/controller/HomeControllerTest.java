package com.JJLin.aiagent.controller;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HomeControllerTest {

    @Test
    void describesAvailableFinanceEndpoints() {
        var response = new HomeController().home();

        assertEquals("UP", response.get("status"));
        assertTrue(response.get("endpoints").toString().contains("POST /api/finance/chat"));
    }
}
