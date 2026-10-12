package com.axiomcloud.service;

import com.axiomcloud.model.UserEntity;
import com.axiomcloud.repository.UserRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.time.OffsetDateTime;
import java.util.Map;

@Service
public class UserService {

    private final UserRepository userRepository;

    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Transactional
    public Map<String, Object> registerOrUpdateUser(String firebaseUid, String email, String displayName) {
        UserEntity user = userRepository.findByFirebaseUid(firebaseUid)
                .orElse(null);

        if (user != null) {
            user.setLastLogin(OffsetDateTime.now());
            if (displayName != null && !displayName.isBlank()) {
                user.setDisplayName(displayName);
            }
            userRepository.save(user);
            return Map.of("id", user.getId(), "email", user.getEmail(), "status", "updated");
        } else {
            UserEntity newUser = new UserEntity();
            newUser.setFirebaseUid(firebaseUid);
            newUser.setEmail(email);
            newUser.setDisplayName(displayName);
            newUser.setLastLogin(OffsetDateTime.now());
            UserEntity saved = userRepository.save(newUser);
            return Map.of("id", saved.getId(), "email", saved.getEmail(), "status", "created");
        }
    }

    public Map<String, Object> getUser(String firebaseUid) {
        UserEntity user = userRepository.findByFirebaseUid(firebaseUid)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));

        return Map.of(
                "id", user.getId(),
                "email", user.getEmail(),
                "display_name", user.getDisplayName() != null ? user.getDisplayName() : "",
                "created_at", user.getCreatedAt() != null ? user.getCreatedAt().toString() : "",
                "last_login", user.getLastLogin() != null ? user.getLastLogin().toString() : ""
        );
    }
}
