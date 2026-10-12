package com.axiomcloud;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
public class AxiomCloudApplication {

    public static void main(String[] args) {
        SpringApplication.run(AxiomCloudApplication.class, args);
    }
}
