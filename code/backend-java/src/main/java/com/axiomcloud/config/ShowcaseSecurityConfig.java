package com.axiomcloud.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Profile;
import org.springframework.http.HttpMethod;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configurers.AbstractHttpConfigurer;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;

@Configuration
@EnableWebSecurity
@Profile("showcase")
public class ShowcaseSecurityConfig {

    private final ShowcaseAuthenticationFilter showcaseAuthenticationFilter;

    public ShowcaseSecurityConfig(ShowcaseAuthenticationFilter showcaseAuthenticationFilter) {
        this.showcaseAuthenticationFilter = showcaseAuthenticationFilter;
    }

    @Bean
    public SecurityFilterChain showcaseSecurityFilterChain(HttpSecurity http) throws Exception {
        CookieCsrfTokenRepository csrfTokens = CookieCsrfTokenRepository.withHttpOnlyFalse();
        http
                .csrf(csrf -> csrf
                        .csrfTokenRepository(csrfTokens)
                        .ignoringRequestMatchers("/api/showcase/session"))
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                        .requestMatchers("/api/health", "/api/showcase/session").permitAll()
                        .anyRequest().authenticated())
                .addFilterBefore(showcaseAuthenticationFilter, UsernamePasswordAuthenticationFilter.class);
        return http.build();
    }
}
