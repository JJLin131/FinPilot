package com.JJLin.aiagent;

import com.JJLin.aiagent.config.CapabilityProperties;
import com.JJLin.aiagent.config.WorkflowProperties;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
@MapperScan("com.JJLin.aiagent.mapper")
@ConfigurationPropertiesScan(basePackageClasses = {CapabilityProperties.class, WorkflowProperties.class})
public class AiAgentApplication {

    public static void main(String[] args) {
        SpringApplication.run(AiAgentApplication.class, args);
    }
}