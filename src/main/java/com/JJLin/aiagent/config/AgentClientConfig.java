package com.JJLin.aiagent.config;

import com.JJLin.aiagent.client.EcommerceHttpClient;
import com.JJLin.aiagent.client.BrowserClient;
import com.JJLin.aiagent.client.DesktopBrowserClient;
import com.JJLin.aiagent.client.WorkspaceFileClient;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.deepseek.DeepSeekChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;

@Configuration
public class AgentClientConfig {

    @Bean(name = "orchestratorAgent")
    public ChatClient orchestratorAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You are the orchestrator agent for a standalone e-commerce AI service. "
                        + "You are responsible for routing, planning, and final response composition. "
                        + "You do not directly access downstream services.")
                .build();
    }

    @Bean(name = "planAgent")
    public ChatClient planAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You are the planning agent. "
                        + "Produce structured intent-level plans only. "
                        + "Describe what should happen, why it matters, and what information is still needed. "
                        + "Do not choose handlers, services, action types, or concrete parameter names.")
                .build();
    }

    @Bean(name = "executorAgent")
    public ChatClient executorAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You are the execution planning agent. "
                        + "Convert an intent-level plan into executable actions using only the supported capabilities provided in context. "
                        + "Choose the most direct supported capability, use only allowed parameter names, and never invent handlers or params. "
                        + "If business information is missing, return blocked=true with missingInfo instead of fabricating values.")
                .build();
    }

    @Bean(name = "verifyAgent")
    public ChatClient verifyAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You are the verification agent. "
                        + "Validate plan completeness and execution evidence. "
                        + "Do not rewrite the final user response.")
                .build();
    }

    @Bean(name = "profileExtractorAgent")
    public ChatClient profileExtractorAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You extract structured user profile fields from a single user message. "
                        + "Return only fields the user explicitly stated in this message. "
                        + "Never infer missing profile fields. "
                        + "Return structured JSON matching the target schema. "
                        + "If no explicit profile fields are present, return an empty object.")
                .build();
    }

    @Bean(name = "longTermMemoryExtractorAgent")
    public ChatClient longTermMemoryExtractorAgent(DeepSeekChatModel model) {
        return ChatClient.builder(model)
                .defaultSystem("You extract long-term user memory from a single user message. "
                        + "Return a structured list of durable user memories only. "
                        + "You may summarize explicit preferences, budget constraints, gift/usage scenarios, and user background expressed in the message. "
                        + "Do not include system/tool failures. "
                        + "Do not repeat structured profile fields that already exist in the provided profile context. "
                        + "If nothing is worth storing, return an empty list.")
                .build();
    }

    @Bean
    public EcommerceHttpClient ecommerceHttpClient(CapabilityProperties properties) {
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout((int) properties.getConnectTimeout().toMillis());
        requestFactory.setReadTimeout((int) properties.getReadTimeout().toMillis());

        RestClient.Builder builder = RestClient.builder()
                .baseUrl(properties.getBaseUrl())
                .requestFactory(requestFactory);
        properties.getDefaultHeaders().forEach((name, value) -> {
            if (name != null && !name.isBlank() && value != null && !value.isBlank()) {
                builder.defaultHeader(name, value);
            }
        });
        if (properties.getAuthHeaderName() != null
                && !properties.getAuthHeaderName().isBlank()
                && properties.getAuthHeaderValue() != null
                && !properties.getAuthHeaderValue().isBlank()) {
            builder.defaultHeader(properties.getAuthHeaderName(), properties.getAuthHeaderValue());
        }
        return new EcommerceHttpClient(builder.build());
    }

    @Bean
    public WorkspaceFileClient workspaceFileClient(CapabilityProperties properties) {
        return new WorkspaceFileClient(properties);
    }

    @Bean
    public BrowserClient browserClient() {
        return new DesktopBrowserClient();
    }
}
